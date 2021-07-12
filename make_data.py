#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 20:48:51 2021

@author: rpa
"""

import os
import glob

import torchvision

from PIL import Image, ImageFilter


DATA_PATH = r"/home/rpa/DL_Data/ImageNet/imagenetv2"

#OUTPUT_PATH = r"/home/rpa/DL_Data/SuperRes/dataset_1"
#MAX_INPUT_FILES = 1000
#STRIDE = 14

OUTPUT_PATH = r"/home/rpa/DL_Data/SuperRes/dataset_2"
MAX_INPUT_FILES = 400000
STRIDE = 33

from SRCNNDataset import IMG_WIDTH
from SRCNNDataset import UPSCALING_FACTOR

# number of all files = 499606

def downsample(img):
    w = img.size[0]
    h = img.size[1]    
    #img = img.filter(ImageFilter.GaussianBlur(radius = 3))    
    resize_down = torchvision.transforms.Resize((int(h / UPSCALING_FACTOR), int(w / UPSCALING_FACTOR)))
    resize_up = torchvision.transforms.Resize((h, w))
    img = resize_down(img)
    img = resize_up(img)
    return img


def extract_tiles(img, sx, sy, stride):
    tiles = []
    w = img.size[0]
    h = img.size[1]    
    for i in range(0, w-sx, stride):
        for j in range(0, h-sy, stride):
            bbox = (i, j, i + sx, j + sy)
            tile = img.crop(bbox)
            tiles.append(tile)
    return tiles


def process_image(img_name, index):
    img_h = Image.open(img_name);
    img_h_ycbcr = img_h.convert('YCbCr')
    img_h_y, img_h_cb, img_h_cr = img_h_ycbcr.split()
        
    img_l = downsample(img_h)
    img_l_ycbcr = img_l.convert('YCbCr')
    img_l_y, img_l_cb, img_l_cr = img_l_ycbcr.split()
    
    tiles_h = extract_tiles(img_h_y, IMG_WIDTH, IMG_WIDTH, STRIDE)
    tiles_l = extract_tiles(img_l_y, IMG_WIDTH, IMG_WIDTH, STRIDE)

    for i in range(len(tiles_h)):
        name_h = "tile" + str(index + i) + "h.png"
        name_l = "tile" + str(index + i) + "l.png"
        tile_h = tiles_h[i];
        tile_l = tiles_l[i];
        tile_h.save(OUTPUT_PATH + os.sep + name_h)
        tile_l.save(OUTPUT_PATH + os.sep + name_l)
    
    return index + len(tiles_h)


# make output dir
try:
    os.mkdir(OUTPUT_PATH)
except IOError:
    pass    

# remove existing files in output dir
for fl in glob.glob(OUTPUT_PATH + os.sep + "*.png"):
    os.remove(fl)

# process number of files
index = 0

#for i in range(0, MAX_INPUT_FILES):
#    name = DATA_PATH + os.sep + "train" + str(i) + ".jpg"
#    print("processing " + "train" + str(i) + ".jpg")
#    if (os.path.exists(name)):
#        index = process_image(name, index)
#    else:
#        break

num_files = 0
for fl in glob.glob(DATA_PATH + os.sep + "*.jpg"):
    index = process_image(fl, index)
    num_files = num_files + 1
    if num_files % 100 == 0 and num_files > 0:
        print("processed " + str(num_files) + " files")
    if num_files > MAX_INPUT_FILES:
        break

print("produced " + str(index) + " tiles")
