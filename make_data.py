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
OUTPUT_PATH = r"/home/rpa/DL_Data/SuperRes"
MAX_INPUT_FILES = 2


def downsample(img):
    h, w = img.shape[:2]
    img2 = cv2.GaussianBlur(img, (3, 3), 10, 10)
    img3 = cv2.resize(img2, (int(h/3), int(w/3)))
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

    tiles_h = extract_tiles(img_h, 33, 33, 14)
    tiles_l = extract_tiles(img_l, 33, 33, 14)

    for i in range(len(tiles_h)):
        name_h = "tile" + str(index + i) + "h.png"
        name_l = "tile" + str(index + i) + "l.png"
        tile_h = tiles_h[i];
        tile_l = tiles_l[i];
        cv2.imwrite(OUTPUT_PATH + os.sep + name_h, tile_h)
        cv2.imwrite(OUTPUT_PATH + os.sep + name_l, tile_l)
    
    return index + len(tiles_h)


# remove existing files in output dir
for fl in glob.glob(OUTPUT_PATH + os.sep + "*.png"):
    os.remove(fl)

# process number of files
index = 0
for i in range(0, MAX_INPUT_FILES):
    name = DATA_PATH + os.sep + "train" + str(i) + ".jpg"
    print("processing " + "train" + str(i) + ".jpg")
    if (os.path.exists(name)):
        index = process_image(name, index)
    else:
        break

print("produced " + str(index) + " tiles")
