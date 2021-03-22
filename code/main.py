import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from sklearn.metrics import confusion_matrix
from dataset import FmriDataset
# from dataset_other_mask import FmriDataset
# from model import Custom3D
from model_lstm import Custom3D
from train_test_set import train_test_subs, params


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


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    k_fold = 5
    accs = []
    cf_matrix = []
    for k in range(k_fold):
        train_subs, test_subs = train_test_subs(test_pct=0.2)
        # print(train_subs)
        # print(test_subs)
        params.update({'subs': train_subs})
        trainset = FmriDataset(params=params, img_timesteps=85)
        params.update({'subs': test_subs})
        testset = FmriDataset(params=params, img_timesteps=85)

        class_weights = torch.FloatTensor(
            [trainset.class_weights[i] for i in range(params.nClass)]).to(device)
        # Initialize the model
        net = Custom3D(params=params).to(device)
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
