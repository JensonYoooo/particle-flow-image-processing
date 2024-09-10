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

from tqdm import tqdm

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
        self.conv_0_2 = VGGBlock(64 ,16, 16)
        self.conv_1_2 = VGGBlock(48, 8, 8)
        
        #col 3
        self.conv_0_3 = VGGBlock(24, 8, 8)
        
        
        
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
        x_0_2 = self.conv_0_2(torch.cat([x_0_1, self.up(x_1_1)], dim = 1))
        
        x_3_0 = self.conv_3_0(self.pool(x_2_0))
        x_2_1 = self.conv_2_1(torch.cat([x_2_0, self.up(x_3_0)], dim = 1))
        x_1_2 = self.conv_1_2(torch.cat([x_1_1, self.up(x_2_1)], dim = 1))
        x_0_3 = self.conv_0_3(torch.cat([x_0_2, self.up(x_1_2)], dim = 1))
        
       
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






# U-net training 

#defining 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = Nested_U_net(4, 1, deep_supervision = True).to(device)
loss_func = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(network.parameters(), lr = 0.01)



output_loss_1 = []
output_loss_2 = []
output_loss_3 = []

for epoch in range(80):
    for count, (data, mask) in enumerate(tqdm(train_loader)):
        mask = mask.resize(data.size(0), 1, 56, 56)   
        optimizer.zero_grad()
        out = network(data)
        if getattr(network, "deep_supervision"):
            tot_loss = sum([loss_func(i, mask) for i in out])
            loss1 = loss_func(out[0], mask)
            loss2 = loss_func(out[1], mask)
            loss3 = loss_func(out[2], mask)
        else:
            tot_loss = loss_func(out[-1], mask)
            loss1, loss2, loss3 = None, None, None
            
        if (loss1, loss2, loss3) != (None, None, None):
            output_loss_1.append(loss1.item())
            output_loss_2.append(loss2.item())
            output_loss_3.append(loss3.item())
            
        tot_loss.backward()
        optimizer.step()
     print(f"epochs: {epoch}, overall loss: {tot_loss.item()}, losses:{loss1.item(), loss2.item(), loss3.item()} 




# convolutional processor training

processor = Output_processor().to(device)
criteria = nn.MSELoss().to(device)
optim = torch.optim.Adam(processor.parameters(), lr = 0.01)

for epoch in range(20):
    for count, (data, mask) in enumerate(tqdm(train_loader)):
        mask = mask.resize(data.size(0), 1, 56, 56)   
        optim.zero_grad()
        network_out = torch.stack(network(data), dim = 1)
       
        out = processor(network_out.resize(network_out.size(0), 3, 56, 56))
        
        loss = criteria(out, mask) 
        loss.backward()
        optim.step()
        
    print(f"epochs: {epoch},loss: {loss.item()}  lr:{optim.param_groups[0]['lr']}")

#model evaluation

best_output = torch.argmin(torch.tensor([output_loss_1[-1], output_loss_2[-1], output_loss_3[-1]]))
from torcheval.metrics.functional import r2_score
Rms = lambda x : np.sqrt(np.mean(x**2))


#with mean outputs
with torch.no_grad():
    tot_rms = 0
    count = 0
    for c, (data, target) in enumerate(test_loader):
        
        if c<5:
            for i in range(data.size(0)):

                inputs = data[i].resize(1, 4, 56, 56)
                out = torch.mean(torch.stack(network(inputs)), dim = 0).reshape(56, 56).flatten().numpy()

                truth = target[i].reshape(56,56).flatten().numpy()
                error = out - truth
                tot_rms += Rms(error)
                count += 1
    print(f"average rms for first 500 images in test loader: {tot_rms/count}")



#with last output

with torch.no_grad():
    tot_rms = 0
    count = 0
    for c, (data, target) in enumerate(test_loader):
    
        if c<5:
            for i in range(data.size(0)):

                inputs = data[i].resize(1, 4, 56, 56)
                out = network(inputs)[-1].reshape(56, 56).flatten().numpy()

                truth = target[i].reshape(56,56).flatten().numpy()
                error = out - truth
                tot_rms += Rms(error)
                count += 1
    print(f"average rms for first 500 images in test loader: {tot_rms/count}")

#with selected output
with torch.no_grad():
    tot_rms = 0
    count = 0
    for c, (data, target) in enumerate(test_loader):
        
        if c<5:
            for i in range(data.size(0)):

                inputs = data[i].resize(1, 4, 56, 56)
                out = network(inputs)[best_output].reshape(56, 56).flatten().numpy()

                truth = target[i].reshape(56,56).flatten().numpy()
                error = out - truth
                tot_rms += Rms(error)
                count += 1
    print(f"average rms for first 500 images in test loader: {tot_rms/count}")


    #with combined output
with torch.no_grad():
    tot_rms = 0
    count = 0
    for c, (data, target) in enumerate(test_loader):
        
        if c<5:
            for i in range(data.size(0)):

                inputs = data[i].resize(1, 4, 56, 56)
                
                out = torch.stack(network(inputs)).resize(3, 56, 56)
                out = processor(out).reshape(56, 56).flatten().numpy()
                
                truth = target[i].reshape(56,56).flatten().numpy()
                error = out - truth
                tot_rms += Rms(error)
                count += 1
    print(f"average rms for first 500 images in test loader: {tot_rms/count}")

    network.eval()
with torch.no_grad():
    for count, (data, target) in enumerate(test_loader):
        inputs = data[count].resize(1, 4, 56, 56)
        
        output = processor(torch.stack(network(inputs)).resize(1, 3, 56, 56)).reshape(56, 56)
        target = target[count]
        break
        
    plt.imshow(output)
    plt.title('predicted image')
    plt.show()
    plt.imshow(target)
    plt.title('truth image')
    plt.show()



plt.hist2d(target.numpy().flatten(),output.numpy().flatten(), bins = (50,50), vmax = 10, cmap = "Blues")
plt.xlabel("truth")
plt.ylabel("predicted")
plt.text(7, 12,f"R^2 score:{r2_score(output, target)}")
plt.title(f"correlation between truth and predicted pixels, test image{1}")
plt.show()



error = output.numpy().flatten() - target.numpy().flatten()
h, xedges, yedges = plt.hist(error, bins = 100)



plt.text(0.2,max(h),f"RMS:{Rms(error)}")
plt.text(0.2,max(h)-80,f"std:{np.std(error)}")
plt.xlabel("predicted-truth")
plt.ylabel("frequency")
plt.show()
    






