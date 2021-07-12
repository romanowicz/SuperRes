"""
Created on Mon Jul 12 14:08:45 2021

@author: rpa
"""

import os
import glob
import torch

from torch.utils.data import Dataset
from torchvision.io import read_image


IMG_WIDTH = 33


class SRCNNDataset(Dataset):
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
        image_l = image_l.type(torch.FloatTensor)
        image_h = image_h.type(torch.FloatTensor)
        if self.transform:
            image_l = self.transform(image_l)
        if self.target_transform:
            image_h = self.target_transform(image_h)
        return image_l, image_h

