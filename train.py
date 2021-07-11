#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 22:06:56 2021

@author: rpa
"""

#import torch
import os
import glob

from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader
#from torchvision import datasets
#from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt


DATA_PATH = r"/home/rpa/DL_Data/SuperRes"

batch_size = 64


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.num_tiles = int(len(glob.glob(img_dir + os.sep + "*.png")) / 2)
        self.img_labels = [0] * self.num_tiles
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.num_tiles

    def __getitem__(self, idx):
        img_path_l = os.path.join(self.img_dir, "tile" + str(idx) + "l.png")
        img_path_h = os.path.join(self.img_dir, "tile" + str(idx) + "h.png")
        image_l = read_image(img_path_l)
        image_h = read_image(img_path_h)
        if self.transform:
            image_l = self.transform(image_l)
        if self.target_transform:
            image_h = self.target_transform(label_h)
        return image_l, image_h


# make dataset and loader
dataset = CustomImageDataset("", DATA_PATH)
train_dataloader = DataLoader(dataset, batch_size=batch_size)

# print info
for X, Y in train_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape, X.dtype)
    print("Shape of Y [N, C, H, W]: ", y.shape, y.dtype)
    break



