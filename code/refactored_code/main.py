import argparse
import math
import os
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchio as tio
from torch import optim
from sklearn.metrics import confusion_matrix

from model import FmriModel
from config import params, split_train_val
from dataset import FmriDataset



def get_confusion_matrix(params, preds, actual):
    preds = [int(k) for k in preds]
    actual = [int(k) for k in actual]
    
    cf = confusion_matrix(actual, preds, labels=list(range(params.num_classes)))
    return cf

def test(model, data_loader):
    correct, total = 0, 0
    preds, actual = [], []
    
    with torch.no_grad():
        for batch in tqdm(data_loader):
            if not batch:
                continue
            inputs, labels =  batch[0].to(params.device), batch[1].to(params.device)
            outputs = model(inputs)
            _, class_pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (class_pred == labels).sum().item()
            preds.extend(list(class_pred.to(dtype=torch.int64)))
            actual.extend(list(labels.to(dtype=torch.int64)))
            
    acc = 100*(correct/total)
    return preds, actual, acc

def train(model, train_loader, val_loader, params):
    loss_function = nn.CrossEntropyLoss(weight=params.class_weights)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    
    # Star the training
    print(f'Training...')
    for epoch in range(params.num_epochs):
        for batch in tqdm(train_loader):
            inputs, labels = batch[0].to(params.device), batch[1].to(params.device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
        if epoch % 2 != 0:
            # Check train and val accuracy after every two epochs
            _, _, train_acc = test(model, train_loader)
            _, _, val_acc = test(model, val_loader)
            print(f'Epoch: {epoch+1} | Loss: {loss} | Train Acc: {train_acc} | Validation Acc: {val_acc}')
        else:
            print(f'Epoch: {epoch+1} | Loss: {loss}')
    
    print('Training complete')
    return model


if __name__ == '__main__':
    # Hyperparameters settings
    parser = argparse.ArgumentParser(description='fMRI training for fatigue prediction hyperparameters')
    parser.add_argument('--seg_len', type=int, default=85, help='Number of scans in a segment')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--mask_type', type=str, default='', help='Type of mask to be used')
    args = parser.parse_args()
    if args.mask_type:
        params.mask_type = args.mask_type
        params.include_mask = True
        print(f'Using {params.mask_type} to train the model')
    params.num_epochs = args.epochs
    params.seg_len = args.seg_len
    params.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Specify the types of transforms to be applied to the fMRI scans
    spatial_transforms = {
        tio.RandomElasticDeformation(): 0.2,
        tio.RandomAffine(): 0.8
    }
    transform = tio.Compose([
        #tio.OneOf(spatial_transforms, p=0.5),
        tio.RandomAffine(),
        tio.ZNormalization(),
        tio.RescaleIntensity((0, 1))
    ])
    
    # Split train and validation subjects
    train_subs, val_subs = split_train_val(val_pct=0.2)
    
    # Build the training set
    params.update({'current_subs': train_subs})
    train_set = FmriDataset(params=params, transform=transform)
    
    # Build the validation set
    params.update({'current_subs': val_subs})
    val_set = FmriDataset(params=params, transform=transform)
    
    params.class_weights = torch.FloatTensor(
        [train_set.class_weights[i] for i in range(params.num_classes)]
    ).to(params.device)
    
    # Initialize the model
    model = FmriModel(params=params).to(params.device)
    
    # DataParallel settings
    params.num_gpus = torch.cuda.device_count()
    print(f'Number of GPUs available: {params.num_gpus}')
    if params.device.type == 'cuda' and params.num_gpus > 1:
        model = nn.DataParallel(model, list(range(params.num_gpus)))
    
    # Load train and validation sets
    train_loader = DataLoader(train_set, batch_size=params.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=params.batch_size, shuffle=True)
    
    # Train the model
    model = train(model, train_loader, val_loader, params)
    
    # Once trained, save the model checkpoint
    current_time = datetime.now().strftime('%m_%d_%Y_%H_%M')
    torch.save(model.state_dict(), f'{current_time}-lr-{params.learning_rate}-epochs-{params.num_epochs}.pth')
    
    # Validate the model
    preds, actual, acc = test(model, val_loader)
    print(f'Validation Accuracy: {acc}')
    print(get_confusion_matrix(params, preds, actual))
    
    # Also, print the train and val subs for information
    print(f'Train subs: {train_subs}\n\nValidation subs: {val_subs}')
    
    
