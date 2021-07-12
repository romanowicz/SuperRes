#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 20:48:51 2021

@author: rpa
"""

import os
import shutil
import glob
import sys

import torchvision

from PIL import Image, ImageFilter


# number of all files = 499606
INPUT_DATA_PATH = r"/home/rpa/DL_Data/ImageNet/imagenetv2"

from SRCNNDataset import IMG_WIDTH
from SRCNNDataset import UPSCALING_FACTOR


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


def process_image(img_name, index, output_path, stride):
    img_h = Image.open(img_name);
    img_h_ycbcr = img_h.convert('YCbCr')
    img_h_y, img_h_cb, img_h_cr = img_h_ycbcr.split()
        
    img_l = downsample(img_h)
    img_l_ycbcr = img_l.convert('YCbCr')
    img_l_y, img_l_cb, img_l_cr = img_l_ycbcr.split()
    
    tiles_h = extract_tiles(img_h_y, IMG_WIDTH, IMG_WIDTH, stride)
    tiles_l = extract_tiles(img_l_y, IMG_WIDTH, IMG_WIDTH, stride)

    for i in range(len(tiles_h)):
        name_h = "tile" + str(index + i) + "h.png"
        name_l = "tile" + str(index + i) + "l.png"
        tile_h = tiles_h[i];
        tile_l = tiles_l[i];
        
        dir_idx = int((index + i) / 1000)
        try:
            os.mkdir(output_path + os.sep + str(dir_idx))  
        except IOError:
            pass   
        
        tile_h.save(output_path + os.sep + str(dir_idx) + os.sep + name_h)
        tile_l.save(output_path + os.sep + str(dir_idx) + os.sep + name_l)
    
    return index + len(tiles_h)


def make_dataset(input_data_path, output_data_path, max_files, stride):

    # make empty output directory
    try:
        shutil.rmtree(output_data_path)
    except:
        pass
    
    try:
        os.mkdir(output_data_path)
    except IOError:
        pass        
    
    # process number of files
    index = 0    
    num_files = 0
    for fl in glob.glob(input_data_path + os.sep + "*.jpg"):
        index = process_image(fl, index, output_data_path, stride)
        num_files = num_files + 1
        if num_files % 100 == 0 and num_files > 0:
            print("processed " + str(num_files) + " files")
        if num_files > max_files:
            break
    
    print("produced " + str(index) + " tiles")


def usage():
    print("usage: make_data.py <output_data_path> <max_images> <stride>")
    sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) != 4:    
        usage()

    output_data_path = sys.argv[1]
    num_images = int(sys.argv[2])
    stride = int(sys.argv[3])
    
    make_dataset(INPUT_DATA_PATH, output_data_path, num_images, stride)
    