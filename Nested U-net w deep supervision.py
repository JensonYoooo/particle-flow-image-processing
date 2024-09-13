import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import skimage.io as io
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter



class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(VGGBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class Nested_U_net(nn.Module):
    def __init__(self, in_channels, out_channels, deep_supervision = False):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deep_supervision = deep_supervision
        #encoder path (col0)
        self.conv_0_0 = VGGBlock(in_channels, 8, 8)
        self.conv_1_0 = VGGBlock(8, 16, 16)
        self.conv_2_0 = VGGBlock(16, 32, 32)
        self.conv_3_0 = VGGBlock(32, 64, 64)
        
        #col 1
        self.conv_0_1 = VGGBlock(24, 32, 32)
        self.conv_1_1 = VGGBlock(48, 32, 32)
        self.conv_2_1 = VGGBlock(96, 16, 16)
        
        
        #col 2
        self.conv_0_2 = VGGBlock(8+32+32 ,16, 16)
        self.conv_1_2 = VGGBlock(32+16+16, 8, 8)
        
        #col 3
        self.conv_0_3 = VGGBlock(8+32+16+8, 8, 8)
        
        
        
        #outputs
        self.final_0_1 = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)
        self.final_0_2 = nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1)
        self.final_0_3 = nn.Conv2d(8, out_channels, kernel_size=3, stride=1, padding=1)
       
    def forward(self, x):
        x_0_0 = self.conv_0_0(x)
        x_1_0 = self.conv_1_0(self.pool(x_0_0))
        x_0_1 = self.conv_0_1(torch.cat([x_0_0,self.up(x_1_0)], dim = 1))
        
        x_2_0 = self.conv_2_0(self.pool(x_1_0))
        x_1_1 = self.conv_1_1(torch.cat([x_1_0, self.up(x_2_0)], dim = 1))
        x_0_2 = self.conv_0_2(torch.cat([x_0_1, self.up(x_1_1), x_0_0], dim = 1))
        
        x_3_0 = self.conv_3_0(self.pool(x_2_0))
        x_2_1 = self.conv_2_1(torch.cat([x_2_0, self.up(x_3_0)], dim = 1))
        x_1_2 = self.conv_1_2(torch.cat([x_1_1, self.up(x_2_1), x_1_0], dim = 1))
        x_0_3 = self.conv_0_3(torch.cat([x_0_2, self.up(x_1_2), x_0_0, x_0_1], dim = 1))
        
       
        if self.deep_supervision:
            return [self.final_0_1(x_0_1),
                    self.final_0_2(x_0_2),
                    self.final_0_3(x_0_3)]
            
        else:
            
            return [self.final_0_3(x_0_3)]



class Output_processor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        
        # Set the weights to average the input images
        with torch.no_grad():
            # Each input channel should contribute equally, so set weights to 1/3
            self.conv.weight.data.fill_(1/3)

    def forward(self, x):
        return self.conv(x)
