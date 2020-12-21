from datetime import datetime

import math
import os
import random

import nilearn as nil
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
from torchvision import transforms, utils

from tqdm import tqdm


# Custom import
from train_test_set import train_test_subs, params


class FmriModel(nn.Module):

    def __init__(self, params):
        super(FmriModel, self).__init__()

        self.ndf = params.ndf
        self.nc = 30
        self.nClass = params.nClass

        # Input to the model is (30, 57, 68, 49) <== (t, x, y, z)

        self.conv1 = nn.Sequential(
            nn.Conv2d(params.nX, self.ndf, 5, 2, bias=False),
            nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.ndf*1, self.ndf*2, 5, 2, bias=False),
            nn.BatchNorm2d(self.ndf*2),
            nn.ReLU(True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.ndf*2, self.ndf*4, 5, 2, bias=False),
            nn.BatchNorm2d(self.ndf*4),
            nn.ReLU(True),
        )

        self._to_linear, self._to_lstm = None, None
        x = torch.randn(params.batchSize*self.nc,
                        params.nX, params.nY, params.nZ)
        self.convs(x)

        self.lstm = nn.LSTM(input_size=3840, hidden_size=256,
                            num_layers=1, batch_first=True)

        self.fc1 = nn.Linear(256, self.ndf * 1)

        self.fc2 = nn.Sequential(
            nn.Linear(self.ndf * 1, self.nClass),
        )

    def convs(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self._to_linear is None:
            # First pass: done to know what the output of the convnet is
            self._to_linear = int(x[0].shape[0]*x[0].shape[1]*x[0].shape[2])
            # For LSTM input, divide by batch_size and time_steps (i.e. / by self.nc and 1)
            self._to_lstm = int(self._to_linear/self.nc)

        return x

    def forward(self, x):
        batch_size, timesteps, c, h, w = x.size()
        # Merge batch_size and timesteps into one dimension
        x = x.view(batch_size*timesteps, c, h, w)
        cnn_out = self.convs(x)

        # Prepare the output from CNN to pass through the LSTM layer
        r_in = cnn_out.view(batch_size, timesteps, -1)

        # Flattening is required when we use DataParallel
        self.lstm.flatten_parameters()

        # Get output from the LSTM
        r_out, (h_n, h_c) = self.lstm(r_in)

        # Pass the output of the LSTM to FC layers
        r_out = self.fc1(r_out[:, -1, :])
        r_out = self.fc2(r_out)

        # Apply softmax to the output and return it
        return F.log_softmax(r_out, dim=1)


class FmriDataset(Dataset):

    def __init__(self, params, data_dir='/data/fmri/data', mask_path='/data/fmri/mask/caudate._mask.nii',
                 img_shape=(57, 68, 49, 135), img_timesteps=30):
        self.data_dir, self.params = data_dir, params
        self.img_timesteps = img_timesteps
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
        except Exception as error:
            print(error)
            print(self.samples[idx])
            with open('error.txt', 'a') as error_file:
                error_file.write(str(error) + '\n')
            return None

    def index_data(self):
        """
        Stores all the image_paths with their respective scores/classes in the 
        """
        self.weights = {i: 0 for i in range(self.num_classes)}
        for sub in os.listdir(self.data_dir):
            if sub not in self.params.subs:
                continue
            sub_dir = os.path.join(self.data_dir, sub)
            preproc_dir = os.path.join(sub_dir, f'{sub}.preproc')
            for img_name in os.listdir(preproc_dir):
                img_path = os.path.join(preproc_dir, img_name)
                score = self.get_score(sub_dir, img_name)
                score_class = self.get_class(score)
                # Since we are randomly sampling 15 timesteps from each scan of 135 timesteps,
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
        img = nil.image.load_img(img_path)
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
            weights[key] = max_value / weights[key]

        return weights


def train_test_length(total, test_pct=0.2):
    train_count = int((1-test_pct)*total)
    test_count = total - train_count
    return train_count, test_count


def train(net, train_loader, loss_function, optimizer):
    print('Training...')
    for epoch in range(params.nEpochs):
        for batch in tqdm(train_loader):
            if not batch:
                # Found some corrupted scan files that could not be read
                continue
            inputs, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch} | Loss: {loss}')

    return net


def test(net, test_loader):
    print('Testing...')
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
            class_outputs = net(inputs)
            _, class_prediction = torch.max(class_outputs.data, 1)
            total += labels.size(0)
            correct += (class_prediction == labels).sum().item()
            preds.extend(list(class_prediction.to(dtype=torch.int64)))
            actual.extend(list(labels.to(dtype=torch.int64)))

    acc = 100*correct/total
    print(f'Accuracy: {acc}')
    return preds, actual, acc


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
k_fold = 5
accs = []
cf_matrix = []
for k in range(k_fold):
    train_subs, test_subs = train_test_subs(test_pct=0.2)
    # print(train_subs)
    # print(test_subs)
    params.update({'subs': train_subs})
    trainset = FmriDataset(params=params)
    params.update({'subs': train_subs})
    testset = FmriDataset(params=params)

    class_weights = torch.FloatTensor(
        [trainset.class_weights[i] for i in range(params.nClass)]).to(device)
    # Initialize the model
    net = FmriModel(params=params).to(device)
    # Distributed training on multiple GPUs if available
    n_gpus = torch.cuda.device_count()
    print(f'Number of GPUs available: {n_gpus}')
    if (device.type == 'cuda') and (n_gpus > 1):
        net = nn.DataParallel(net, list(range(n_gpus)))

    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train_loader = DataLoader(
        trainset, batch_size=params.batchSize, shuffle=True)
    test_loader = DataLoader(
        testset, batch_size=params.batchSize, shuffle=True)

    net = train(net, train_loader, loss_function, optimizer)
    # Save the model checkpoint
    current_time = datetime.now()
    current_time = current_time.strftime("%m%d%Y%H_%M")
    torch.save(net.state_dict(), f'{current_time}-scans-5-fold-{k}.pth')
    preds, actual, acc = test(net, test_loader)
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

with open('abc.txt', 'w') as abc_file:
    for cf in cf_matrix:
        abc_file.write(f'{cf}\n')
    abc_file.write(f'{accs}')
