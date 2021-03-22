import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.autograd as dif
from torch.nn.modules.utils import _triple


""" -------------------------------------------------------------------------"""
# R2Plus1D Convolution
class SpatioTemporalConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input 
    planes with distinct spatial and time axes, by performing a 2D convolution over the 
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time 
    axis to produce the final output.

    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
        spatial_stride =  [1, stride[1], stride[2]]
        spatial_padding =  [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride =  [stride[0], 1, 1]
        temporal_padding =  [padding[0], 0, 0]

        # compute the number of intermediary channels (M) using formula 
        # from the paper section 3.5
        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/ \
                            (kernel_size[1]* kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        # the spatial conv is effectively a 2D conv due to the 
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                    stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU()

        # the temporal conv is effectively a 1D conv, but has batch norm 
        # and ReLU added inside the model constructor, not here. This is an 
        # intentional design choice, to allow this module to externally act 
        # identical to a standard Conv3D, so it can be reused easily in any 
        # other codebase
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size, 
                                    stride=temporal_stride, padding=temporal_padding, bias=bias)

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x




""" Classifier """
class Custom3D(nn.Module):
    def __init__(self, params):
        super(Custom3D, self).__init__()
        self.ndf = params.ndf
        # self.nc = params.nT // params.nDivT
        self.nc = 30
        self.nClass = params.nClass
        # print(params.nX, params.nY, params.nZ) ==> 57, 68, 49
        
        self.conv1 = nn.Sequential(
            ## input is 15 x 54 x 64 x 50
            nn.Conv2d(params.nX, self.ndf, 5, 2, bias = False),
            nn.ReLU(True),
            ## state size. (ndf) x 26 x 31 x 24
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.ndf*1, self.ndf*2, 5, 2, bias = False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.ReLU(True),
            ## state size. (ndf*2) x 13 x 15 x 12
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.ndf*2, self.ndf*4, 5, 2, bias = False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.ReLU(True),
            ## state size. (ndf*4) x 6 x 7 x 6
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.ndf*4, self.ndf*4, 4, 2, bias = False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.ReLU(True),
            ## state size. (ndf*2) x 3 x 3 x 3
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(self.ndf*4, self.ndf*2, 3, 1, bias = False),
            nn.ReLU(True),
        )
        
        self._to_linear, self._to_lstm = None, None
        x = torch.randn(params.batchSize*self.nc, params.nX, params.nY, params.nZ)
        self.convs(x)
        
        self.lstm = nn.LSTM(input_size=1920, hidden_size=256, num_layers=1, batch_first=True)
        
        self.fc1 = nn.Linear(256, self.ndf * 1)

        
        self.fc2 = nn.Sequential(
            nn.Linear(self.ndf * 1, self.nClass),
        )
        
    def convs(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        
        # This is to make sure that we don't have to worry about the shape from the convolutional layers
        # before sending the input to the FC layers
        if self._to_linear is None:
            self._to_linear = int(x[0].shape[0]*x[0].shape[1]*x[0].shape[2])
            # For LSTM input, divide by batch_size and time_steps (i.e. / by self.nc and 1)
            self._to_lstm = int(self._to_linear/self.nc)
            
        return x
    
    
    def forward(self, x):
        batch_size, timesteps, c, h, w = x.size()
        x = x.view(batch_size*timesteps, c, h, w)
        cnn_out = self.convs(x)
        r_in = cnn_out.view(batch_size, timesteps, -1)
        self.lstm.flatten_parameters()
        r_out, (h_n, h_c) = self.lstm(r_in)
        r_out = self.fc1(r_out[:, -1, :])
        r_out = self.fc2(r_out)
        return F.log_softmax(r_out, dim=1)