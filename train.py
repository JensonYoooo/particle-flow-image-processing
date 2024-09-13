import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split


def U_net_train(network, train_loader, loss_func, optimizer, epochs = 200):
    output_loss_1 = []
    output_loss_2 = []
    output_loss_3 = []
  
    for epoch in range(epochs):
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
            
        print(f"epochs: {epoch}, overall loss: {tot_loss.item()}, losses:{loss1.item(), loss2.item(), loss3.item()} lr:{optimizer.param_groups[0]['lr']}")
    return network, output_loss_1, output_loss_2, output_loss_3


def processor_train(network, processor, train_loader, test_loader, criteria, optimizer, epochs = 20):
    '''note: the network must be trained prior to teh processor, as the processor only deal with the outcomes'''
    for epoch in range(epochs):
        for count, (data, mask) in enumerate(tqdm(train_loader)):
            mask = mask.resize(data.size(0), 1, 56, 56)   
            optim.zero_grad()
            network_out = torch.stack(network(data), dim = 1)
           
            out = processor(network_out.resize(network_out.size(0), 3, 56, 56))
            
            loss = criteria(out, mask) 
            loss.backward()
            optim.step()
            
        print(f"epochs: {epoch},loss: {loss.item()}  lr:{optim.param_groups[0]['lr']}")
  return processor
                
