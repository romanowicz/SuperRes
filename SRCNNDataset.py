"""
Created on Mon Jul 12 14:08:45 2021

@author: Roman Parys (romanowicz@protonmail.ch)
"""

import os
import glob

from PIL import Image
#from PIL import ImageFilter

from torch.utils.data import Dataset
import torchvision


TILE_SIZE = 33
STRIDE = 14
IMG_SIZE_X = 192
IMG_SIZE_Y = 192
UPSCALING_FACTOR = 3


class SRCNNDataset(Dataset):

    def downsample(self, img):
        w = img.size[0]
        h = img.size[1]    
        #img = img.filter(ImageFilter.GaussianBlur(radius = 3))    
        resize_down = torchvision.transforms.Resize((int(h / UPSCALING_FACTOR), int(w / UPSCALING_FACTOR)))
        resize_up = torchvision.transforms.Resize((h, w))
        img = resize_down(img)
        img = resize_up(img)
        return img
        
    
    def tilesPerImage(self, image_w, tile_w, stride):
        num = 0
        for i in range(0, image_w - tile_w, stride):
            num = num + 1
        return num


    def __init__(self, annotations_file, img_dir, max_images=None, transform=None, target_transform=None):        
        self.file_list = glob.glob(img_dir + os.sep + "*.jpg")
        if max_images is not None and max_images > 0:
            m = min(max_images, len(self.file_list))
            self.file_list = self.file_list[:m]

        self.img_dir = img_dir
        self.tiles_x = self.tilesPerImage(IMG_SIZE_X, TILE_SIZE, STRIDE)
        self.tiles_y = self.tilesPerImage(IMG_SIZE_Y, TILE_SIZE, STRIDE)
        self.num_tiles_per_img = self.tiles_x * self.tiles_y
        self.num_tiles = self.num_tiles_per_img * len(self.file_list)
        
        print("Dataset info")
        print("num_images    : " + str(len(self.file_list)))
        print("num_tiles     : " + str(self.num_tiles))

        self.img_h = None
        self.img_l = None
        self.img_idx = -1

        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return self.num_tiles


    def __getitem__(self, idx):
        
        img_idx_curr = int(idx / self.num_tiles_per_img)

        if img_idx_curr != self.img_idx:
            # load images
            img_name = self.file_list[img_idx_curr]
            img_h_rgb = Image.open(img_name);
            img_h_ycbcr = img_h_rgb.convert('YCbCr')
            self.img_h, img_h_cb, img_h_cr = img_h_ycbcr.split()                

            img_l_rgb = self.downsample(img_h_rgb)
            img_l_ycbcr = img_l_rgb.convert('YCbCr')
            self.img_l, img_l_cb, img_l_cr = img_l_ycbcr.split()
            
            self.img_idx = img_idx_curr
                
        tile_idx = idx - self.img_idx * self.num_tiles_per_img
        tile_idx_x = int(tile_idx % self.tiles_x)
        tile_idx_y = int(tile_idx / self.tiles_x)                
        tile_offs_x = tile_idx_x * STRIDE
        tile_offs_y = tile_idx_y * STRIDE
        bbox = (tile_offs_x, tile_offs_y, tile_offs_x + TILE_SIZE, tile_offs_y + TILE_SIZE)
        tile_l = self.img_l.crop(bbox)
        tile_h = self.img_h.crop(bbox)
        
        if self.transform:
            tile_l = self.transform(tile_l)
        if self.target_transform:
            tile_h = self.target_transform(tile_h)        
        
        # debug
        if False:
            print("-----------------------------")
            print("idx       : " + str(idx))
            print("img_idx   : " + str(img_idx_curr))
            print("img_idx   : " + str(self.img_idx))
            print("tile_idx   : " + str(tile_idx))
            print("tile_idx_x : " + str(tile_idx_x))
            print("tile_idx_y : " + str(tile_idx_y))
            try:
                os.mkdir("tmp")  
            except IOError:
                pass   
            name_l = "tmp" + os.sep + "tile_" + str(idx) + "_l.png"
            name_h = "tmp" + os.sep + "tile_" + str(idx) + "_h.png"
            to_image = torchvision.transforms.ToPILImage()
            to_image(tile_l).save(name_l)
            to_image(tile_h).save(name_h)
            
        return tile_l, tile_h

