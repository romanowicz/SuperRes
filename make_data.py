#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 20:48:51 2021

@author: rpa
"""

import os
import glob
import cv2


DATA_PATH = r"/home/rpa/DL_Data/ImageNet/imagenetv2"

#OUTPUT_PATH = r"/home/rpa/DL_Data/SuperRes/dataset_0"
#MAX_INPUT_FILES = 1000
#STRIDE = 14

OUTPUT_PATH = r"/home/rpa/DL_Data/SuperRes/dataset_2"
MAX_INPUT_FILES = 400000
STRIDE = 33

from SRCNNDataset import IMG_WIDTH
from SRCNNDataset import UPSCALING_FACTOR

# number of all files = 499606

def downsample(img):
    h, w = img.shape[:2]
    img2 = cv2.GaussianBlur(img, (3, 3), 10, 10)
    img3 = cv2.resize(img2, (int(h/UPSCALING_FACTOR), int(w/UPSCALING_FACTOR)))
    img4 = cv2.resize(img3, (h, w))
    return img4


def extract_tiles(img, sx, sy, stride):
    tiles = []
    h, w = img.shape[:2]
    for i in range(0, w-sx, stride):
        for j in range(0, h-sy, stride):
            tile = img[j:j+sy, i:i+sx]
            tiles.append(tile)
    return tiles


def process_image(img_name, index):
    img_h = cv2.imread(img_name)
    img_l = downsample(img_h)

    tiles_h = extract_tiles(img_h, IMG_WIDTH, IMG_WIDTH, STRIDE)
    tiles_l = extract_tiles(img_l, IMG_WIDTH, IMG_WIDTH, STRIDE)

    for i in range(len(tiles_h)):
        name_h = "tile" + str(index + i) + "h.png"
        name_l = "tile" + str(index + i) + "l.png"
        tile_h = tiles_h[i];
        tile_l = tiles_l[i];
        cv2.imwrite(OUTPUT_PATH + os.sep + name_h, tile_h)
        cv2.imwrite(OUTPUT_PATH + os.sep + name_l, tile_l)
    
    return index + len(tiles_h)


# make output dir
os.mkdir(OUTPUT_PATH)

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

for fl in glob.glob(DATA_PATH + os.sep + "*.jpg"):
    print("processing " + fl)
    index = process_image(fl, index)

print("produced " + str(index) + " tiles")
