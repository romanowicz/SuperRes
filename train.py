#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 22:06:56 2021

@author: rpa
"""

import sys

import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn

import SRCNNDataset2
import SRCNN


batch_size = 128
num_epochs = 8


def train(data_path, max_images, model_name):

    p = SRCNN.SRCNNParam()
    out_width = SRCNNDataset2.TILE_SIZE - 2 * (int(p.f1 / 2) + int(p.f3 / 2))
    
    # define custom transform function
    transform = torchvision.transforms.Compose([    
        torchvision.transforms.ToTensor()
    ])
    
    target_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.CenterCrop(out_width)
    ])
    
    # make dataset and loader
    #dataset = SRCNNDataset.SRCNNDataset("", data_path, transform=transform, target_transform=target_transform)
    dataset = SRCNNDataset2.SRCNNDataset2("", data_path, max_images=max_images, transform=transform, target_transform=target_transform)
    train_dataloader = DataLoader(dataset, batch_size=batch_size)
    
    # print info
    for X, Y in train_dataloader:
        print("Shape of X [N, C, H, W]: ", X.shape, X.dtype)
        print("Shape of Y [N, C, H, W]: ", Y.shape, Y.dtype)
        break
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    
    model = SRCNN.SRCNN().to(device)
    print(model)
    
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    for epoch in range(0, num_epochs):
        current_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):        
            inputs, targets = data
                
            outputs = model(inputs.to(device))
            targets = targets.to(device)
            
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
            if i % 500 == 499:
              print('Loss after mini-batch %5d: %.3f' %
                    (i + 1, current_loss / 500))
              current_loss = 0.0
    
    print('Training process has finished.')        
      
    # save model
    torch.save(model.state_dict(), model_name)


def usage():
    print("usage: train.py <data_path> <max_images> <model_name>")
    sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) != 4:    
        usage()
        
    data_path = sys.argv[1]
    max_images = int(sys.argv[2])
    model_name= sys.argv[3]
    
    train(data_path, max_images, model_name)
