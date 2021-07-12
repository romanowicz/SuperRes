"""
Created on Mon Jul 12 14:08:45 2021

@author: rpa
"""

import os
import glob

from PIL import Image

from torch.utils.data import Dataset


IMG_WIDTH = 33
UPSCALING_FACTOR = 3


class SRCNNDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):        
        self.num_tiles = int(len(glob.glob(img_dir + os.sep + "*/*.png", recursive=True)) / 2)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.num_tiles

    def __getitem__(self, idx):
        dir_idx = int(idx / 1000)
        img_path_l = os.path.join(self.img_dir, str(dir_idx), "tile" + str(idx) + "l.png")
        img_path_h = os.path.join(self.img_dir, str(dir_idx), "tile" + str(idx) + "h.png")
        image_l = Image.open(img_path_l).convert('L');        
        image_h = Image.open(img_path_h).convert('L');        
        if self.transform:
            image_l = self.transform(image_l)
        if self.target_transform:
            image_h = self.target_transform(image_h)
        return image_l, image_h

