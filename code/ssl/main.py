import argparse
import math
import os
import random
from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchio as tio
from torch import optim
from sklearn.metrics import confusion_matrix

# For distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

# Import MOCO model classes
import moco.loader
import moco.builder

from model import FmriModel
from config import params, split_train_val
from dataset import BoldDataset


"""
Demo Run Command:

>> python main.py --epochs=100 --file_name="affine_motion_noise_caudate"

# file_name is the name that you want to use while saving the model

"""

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0/batch_size))
        return res


def train(train_loader, model, criterion, optimizer, epoch, args):
    # Switch to training model
    model.train()
    
    # Star the training
    for i, segments in enumerate(train_loader):
        segments[0] = segments[0].cuda(args.gpu, non_blocking=True).float()
        segments[1] = segments[1].cuda(args.gpu, non_blocking=True).float()
        
        # Compute output from the model
        output, target = model(im_q=segments[0], im_k=segments[1])
        # print(f'output.shape: {output.shape}')
        # print(f'target: {target}')
        loss = criterion(output, target)
                
        # Compute gradient and do a SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def main_worker(gpu, args):
    # Set GPU id to current GPU to use it for training
    args.gpu = gpu
    # Compute the rank of the current GPU among all (by using node rank and n_gpus_per_node)
    rank = args.nr * args.gpus + gpu
    # Initialize distributed training group
    dist.init_process_group(backend='nccl', init_method='env://',
                           world_size=args.world_size, rank=rank)
    
    # Load model on current GPU using distributed parallel
    print(f'==> Creating model for GPU: {gpu}')
    model = moco.builder.MoCo(FmriModel, args, args.moco_dim, args.moco_k, args.moco_m,
                             args.moco_t, args.mlp)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    model = DistributedDataParallel(model, device_ids=[gpu])
    
    # Data transform: Specify the types of transforms to be applied to the fMRI scans
    spatial_transforms = {
        tio.RandomElasticDeformation(): 0.2,
        tio.RandomAffine(): 0.8
    }
    other_transforms = {
        tio.RandomBlur(): 0.5,
        tio.RandomGamma(): 0.5,
        #tio.RandomNoise(): 0.4
    }
    transform = tio.Compose([
        tio.OneOf(spatial_transforms),
        tio.OneOf(other_transforms),
        tio.RandomMotion(),
        tio.RandomNoise(),
        tio.ZNormalization(),
        tio.RescaleIntensity((0, 1))
    ])
    
    # Load train dataset
    train_dataset = BoldDataset(params=args, transform=transform)
    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                             drop_last=True)
    
    # Criterion and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(args.num_epochs):
        print(f'EPOCH: {epoch}\t', end='')
        train_sampler.set_epoch(epoch)
        train(train_loader, model, criterion, optimizer, epoch, args)
        
        # Save a model checkpoint
        if rank % args.gpus == 0 and epoch == args.num_epochs - 1:
            print(f'Saving model checkpoint at rank: {rank}')
            state = {
                'epochs': epoch,
                'frame_size': f'{args.nX}_{args.nY}_{args.nZ}',
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            filename = f'saved_models/moco_pretrained_{state["frame_size"]}_epoch_{epoch}.pth.tar'
            torch.save(state, filename)

if __name__ == '__main__':
    # Hyperparameters settings
    parser = argparse.ArgumentParser(description='fMRI training for fatigue prediction hyperparameters')
    parser.add_argument('--seg_len', type=int, default=85, help='Number of scans in a segment')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--mask_type', type=str, default='', help='Type of mask to be used')
    parser.add_argument('--file_name', type=str, default='default', help='Saved model file name')
    
    # moco specific configs:
    parser.add_argument('--moco-dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--moco-k', default=65536, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco-m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco-t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    
    # options for moco v2
    parser.add_argument('--mlp', action='store_true',
                        help='use mlp head')
    parser.add_argument('--aug-plus', action='store_true',
                        help='use moco v2 data augmentation')
    parser.add_argument('--cos', action='store_true',
                        help='use cosine lr schedule')
    
    # other arguments
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    
    # Args for distributed training
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=2, type=int, help='number of gpus per node')
    parser.add_argument('-gpu', '--gpu', default=None, type=int, help='GPU id to use for training')
    parser.add_argument('-nr', '--nr', default=0, type=int,help='ranking within the nodes')
    parser.add_argument('-wr', '--workers', default=0, type=int,help='Number of workers')
    
    args = parser.parse_args()
    for key, value in params.items():
        setattr(args, key, value)
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.world_size = args.gpus*args.nodes
    
    # Initialize environment variables for Distributed training
    os.environ['MASTER_ADDR'] = '192.168.88.20'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(main_worker, nprocs=args.gpus, args=(args,))
        
    
