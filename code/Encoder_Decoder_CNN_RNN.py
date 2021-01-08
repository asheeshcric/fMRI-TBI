from datetime import datetime

import math
import os
import random

import nilearn as nil
from nilearn import image as nil_image
import numpy as np

from scipy import ndimage
from sklearn.metrics import confusion_matrix
from PIL import Image

import torch
import torch.autograd as dif
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel

from torch import optim
from torch.nn.modules.utils import _triple
from torch.utils.data import Dataset, DataLoader, random_split
# This is imported to fix any data error in a batch
from torch.utils.data.dataloader import default_collate
from torchvision import transforms, utils

from tqdm import tqdm

# Custom import
from train_test_set import train_test_subs, params


def conv2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
    return outshape


# 2D CNN encoder train from scratch (no transfer learning)
class EncoderCNN(nn.Module):
    def __init__(self, img_x=68, img_y=49, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        super(EncoderCNN, self).__init__()

        self.img_x = img_x
        self.img_y = img_y
        inp_channels = 57
        self.CNN_embed_dim = CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 32, 64, 128, 256
        self.k1, self.k2, self.k3, self.k4 = (
            5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (
            2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (
            0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # conv2D output shapes
        self.conv1_outshape = conv2D_output_size(
            (self.img_x, self.img_y), self.pd1, self.k1, self.s1)  # Conv1 output shape
        self.conv2_outshape = conv2D_output_size(
            self.conv1_outshape, self.pd2, self.k2, self.s2)
        self.conv3_outshape = conv2D_output_size(
            self.conv2_outshape, self.pd3, self.k3, self.s3)
        self.conv4_outshape = conv2D_output_size(
            self.conv3_outshape, self.pd4, self.k4, self.s4)

        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=inp_channels, out_channels=self.ch1,
                      kernel_size=self.k1, stride=self.s1, padding=self.pd1),
            nn.BatchNorm2d(self.ch1, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2,
                      kernel_size=self.k2, stride=self.s2, padding=self.pd2),
            nn.BatchNorm2d(self.ch2, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3,
                      kernel_size=self.k3, stride=self.s3, padding=self.pd3),
            nn.BatchNorm2d(self.ch3, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch3, out_channels=self.ch4,
                      kernel_size=self.k4, stride=self.s4, padding=self.pd4),
            nn.BatchNorm2d(self.ch4, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.drop = nn.Dropout2d(self.drop_p)
        self.pool = nn.MaxPool2d(2)
        # fully connected layer, output k classes
        self.fc1 = nn.Linear(
            self.ch4 * self.conv4_outshape[0] * self.conv4_outshape[1], self.fc_hidden1)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        # output = CNN embedding latent variables
        self.fc3 = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # CNNs
            x = self.conv1(x_3d[:, t, :, :, :])
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = x.view(x.size(0), -1)           # flatten the output of conv

            # FC layers
            x = F.relu(self.fc1(x))
            # x = F.dropout(x, p=self.drop_p, training=self.training)
            x = F.relu(self.fc2(x))
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)
            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=6):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers   # RNN hidden layers
        self.h_RNN = h_RNN                 # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            batch_first=True,
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):

        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        # choose RNN_out at the last time step
        x = self.fc1(RNN_out[:, -1, :])
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x


class FmriDataset(Dataset):

    def __init__(self, params, data_dir='/data/fmri/data', mask_path='/data/fmri/mask/caudate._mask.nii',
                 img_shape=(57, 68, 49, 135)):
        self.data_dir, self.params = data_dir, params
        self.img_timesteps = params.img_timesteps
        self.num_classes = params.nClass
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.mask_path, self.img_shape = mask_path, img_shape
        self.samples = []
        # Initialize the image indexes with their scores
        self.index_data()
        # self.mask = self.read_mask()
        self.class_weights = self.find_weights()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            img_path, score = self.samples[idx]
            score = self.get_class(score)
            img = self.read_image(img_path)
            # img = self.apply_mask(img)
            img = self.apply_temporal_aug(img)
            return img, score
        except Exception:
            return None

    def index_data(self):
        """
        Stores all the image_paths with their respective scores/classes in the 
        """
        self.weights = {i: 0 for i in range(self.num_classes)}
        for sub in os.listdir(self.data_dir):
            if sub not in self.params.subs:
                # Don't consider subjects that are not in the subs set
                continue
            sub_dir = os.path.join(self.data_dir, sub)
            preproc_dir = os.path.join(sub_dir, f'{sub}.preproc')
            for img_name in os.listdir(preproc_dir):
                img_path = os.path.join(preproc_dir, img_name)
                score = self.get_score(sub_dir, img_name)
                score_class = self.get_class(score)
                # Since we are randomly sampling 30 timesteps from each scan of 135 timesteps,
                # I am considering the same image for "n" times so that we have more data to train
                n = 5
                for k in range(n):
                    self.weights[score_class] += 1
                    self.samples.append((img_path, score))

    def get_class(self, score):
        """
        Categorize each score into one of the five classes (bins)
        Returns values from 0-5 (6 classes)
        Classes: (0, 10), (10, 20), (20, 40), (40, 60), (60, 80), (80, 100)
        """
        if score < 10:
            return 0
        elif score >= 10 and score < 20:
            return 1
        elif score >= 20 and score < 40:
            return 2
        elif score >= 40 and score < 60:
            return 3
        elif score >= 60 and score < 80:
            return 4
        else:
            return 5

    def get_score(self, sub_dir, img_name):
        score_file = '0back_VAS-f.1D' if '0back' in img_name else '2back_VAS-f.1D'
        score_path = os.path.join(sub_dir, score_file)
        with open(score_path, 'r') as s_f:
            scores = [int(str(score.replace('\n', ''))) for score in s_f]

        task_num = img_name.split('.')[1]
        score_num = int(task_num[-1:])
        return scores[score_num]

    def read_image(self, img_path):
        nX, nY, nZ, nT = self.img_shape
        img = nil_image.load_img(img_path)
        img = img.get_fdata()[:nX, :nY, :nZ, :nT]
        img = torch.tensor(img, dtype=torch.float, device=self.device)
        img = (img - img.mean()) / img.std()
        return img

    def read_mask(self):
        nX, nY, nZ, _ = self.img_shape
        mask_img = nil.image.load_img(self.mask_path)
        mask_img = mask_img.get_fdata()[:]
        mask_img = np.asarray(mask_img)
        dilated_mask = np.zeros((nX, nY, nZ))
        ratio = round(mask_img.shape[2]/nZ)
        for k in range(nZ):
            temp = ndimage.morphology.binary_dilation(
                mask_img[:, :, k*ratio], iterations=1) * 1
            temp_img = Image.fromarray(np.uint8(temp*255))
            dilated_mask[:, :, k] = np.array(temp_img.resize((nY, nX)))

        dilated_mask = (dilated_mask > 64).astype(int)
        dilated_mask = torch.tensor(
            dilated_mask, dtype=torch.float, device=self.device)
        return dilated_mask

    def apply_mask(self, img):
        nT = img.shape[-1]
        for i in range(nT):
            img[:, :, :, i] = torch.mul(img[:, :, :, i], self.mask)
        return img

    def apply_temporal_aug(self, img):
        """
        Image shape: X, Y, Z, t=135
        So, e.g: take any 30 random timesteps from the 135 available in ascending order 
        """
        total_timesteps = img.shape[3]
        rand_timesteps = sorted(random.sample(
            range(0, total_timesteps), self.img_timesteps))
        # Move time axes to the first place followed by X, Y, Z
        img = img.permute(3, 0, 1, 2)
        img = torch.tensor(np.take(img.cpu().numpy(), rand_timesteps, axis=0))
        return img

    def find_weights(self):
        weights = dict(self.weights)
        key_max = max(weights.keys(), key=(lambda k: weights[k]))
        max_value = weights[key_max]
        for key in weights.keys():
            # Add 1 to the denominator to avoid divide by zero error (in some cases)
            weights[key] = max_value / (weights[key]+1)

        return weights


def my_collate(batch):
    # Function to catch errors while reading a batch of fMRI scans
    # Removes any NoneType values from the batch to prevent errors while training
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def train_test_length(total, test_pct=0.2):
    train_count = int((1-test_pct)*total)
    test_count = total - train_count
    return train_count, test_count


def train(models, train_loader, loss_function, optimizer, test_loader):
    print('Training...')
    cnn_encoder, rnn_decoder = models

    for epoch in range(params.nEpochs):
        for batch in tqdm(train_loader):
            inputs, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()

            outputs = rnn_decoder(cnn_encoder(inputs))
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

        _, _, train_acc = test(models, train_loader)
        _, _, test_acc = test(models, test_loader)

        print(
            f'Epoch: {epoch} | Loss: {loss} | Train Acc: {train_acc} | Test Acc: {test_acc}')

    return [cnn_encoder, rnn_decoder]


def test(models, test_loader):
    cnn_encoder, rnn_decoder = models
    correct = 0
    total = 0

    preds = []
    actual = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            if not data:
                continue
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            class_outputs = rnn_decoder(cnn_encoder(inputs))
            _, class_prediction = torch.max(class_outputs.data, 1)
            total += labels.size(0)
            correct += (class_prediction == labels).sum().item()
            preds.extend(list(class_prediction.to(dtype=torch.int64)))

            actual.extend(list(labels.to(dtype=torch.int64)))

    acc = 100*correct/total
    # print(f'Accuracy: {acc}')
    return preds, actual, acc


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
k_fold = 1
params.nEpochs = 10
accs = []
cf_matrix = []
learning_rate = 0.00001
sample_timesteps = 30
params.update({'img_timesteps': sample_timesteps})

print(f'Parameters: LR: {learning_rate} | Epochs: {params.nEpochs} | K-folds: {k_fold} | BatchSize: {params.batchSize} | Sample timesteps: {sample_timesteps}')


# Get training and testing subjects from the dataset
train_subs, test_subs = train_test_subs(test_pct=0.2)

# Add 'subs' key in the params to track which subjects to use to create both trainset and testset
params.update({'subs': train_subs})
trainset = FmriDataset(params=params)
params.update({'subs': test_subs})
testset = FmriDataset(params=params)

# Identify the training set class weights based on their occurences in the training data
class_weights = torch.FloatTensor(
    [trainset.class_weights[i] for i in range(params.nClass)]).to(device)


for k in range(k_fold):
    # Split the train and validation sets
    train_subs, test_subs = train_test_subs(test_pct=0.2)
    print(f'Train subs: {train_subs} || Test subs: {test_subs}')
    params.update({'subs': train_subs})
    trainset = FmriDataset(params=params)
    params.update({'subs': test_subs})
    testset = FmriDataset(params=params)

    # Get the class weights based on their number of instances
    class_weights = torch.FloatTensor(
        [trainset.class_weights[i] for i in range(params.nClass)]).to(device)

    # Initialize the models
    cnn_encoder = EncoderCNN().to(device)
    rnn_decoder = DecoderRNN().to(device)

    # Distributed training on multiple GPUs if available
    n_gpus = torch.cuda.device_count()
    print(f'Number of GPUs available: {n_gpus}')
    if (device.type == 'cuda') and (n_gpus > 1):
        cnn_encoder = nn.DataParallel(cnn_encoder)
        rnn_decoder = nn.DataParallel(rnn_decoder)

    loss_function = nn.CrossEntropyLoss(weight=class_weights)

    crnn_params = list(cnn_encoder.parameters()) + \
        list(rnn_decoder.parameters())
    optimizer = torch.optim.Adam(crnn_params, lr=learning_rate)

    # Prepare the train and validation loaders
    train_loader = DataLoader(
        trainset, batch_size=params.batchSize, shuffle=True, collate_fn=my_collate)
    test_loader = DataLoader(
        testset, batch_size=params.batchSize, shuffle=True, collate_fn=my_collate)

    # You can use my_collate() function inside the dataloader to check for errors while reading corrupted scans
    models = [cnn_encoder, rnn_decoder]
    models = train(models, train_loader, loss_function, optimizer, test_loader)

    # Save the model checkpoint
#     current_time = datetime.now()
#     current_time = current_time.strftime("%m%d%Y%H_%M")
#     torch.save(net.state_dict(), f'{current_time}-scans-5-fold-{k}.pth')

    # Test the model
    preds, actual, acc = test(models, test_loader)
    accs.append(acc)

    # For confusion matrix
    preds = [int(k) for k in preds]
    actual = [int(k) for k in actual]

    cf = confusion_matrix(actual, preds, labels=list(range(params.nClass)))
    cf_matrix.append(cf)
    with open('cf_matrices.txt', 'a') as cf_file:
        cf_file.write(str(cf) + '\n')


print(cf_matrix)
print(accs)
print(f'Avg Accuracy: {sum(accs)/len(accs)}')
print(f'Parameters: LR: {learning_rate} | Epochs: {params.nEpochs} | K-folds: {k_fold} | BatchSize: {params.batchSize} | Sample timesteps: {sample_timesteps}')

print(f'Train subs: {train_subs} || Test subs: {test_subs}')

with open('abc.txt', 'w') as abc_file:
    for cf in cf_matrix:
        abc_file.write(f'{cf}\n')
    abc_file.write(f'{accs}')
