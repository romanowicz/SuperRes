#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 14:53:58 2021

@author: Roman Parys (romanowicz@protonmail.ch)
"""

import sys
import torch
import torchvision
from PIL import Image

import SRCNN


UPSCALING_FACTOR = 3


def usage():
    print("usage: dump_layer_weights.py <model.pt> <image.png>")
    sys.exit(0)


def get_weight_image(weight):
    to_image = torchvision.transforms.ToPILImage()

    num_filters = weight.shape[0]
    
    filters_x = int(num_filters / 8)
    filters_y = int(num_filters / filters_x)
    h = weight[0][0].shape[0] + 1
    w = weight[0][0].shape[1] + 1
    image = Image.new('RGB', (filters_x * w - 1, filters_y * h - 1), (255, 0, 0))
    
    for i in range(0,weight.shape[0]):
        u = int(i % filters_x)
        v = int(i / filters_x)
        t = weight[i][0]
        mn = torch.min(t)
        mx = torch.max(t)
        t1 = (t - mn) / (mx - mn)
        im = to_image(t1)
        im.copy()
        image.paste(im, (u * w, v * h))
    
    return image


if __name__ == "__main__":
    if len(sys.argv) != 3:
        usage()

    model_file = sys.argv[1]
    image_file = sys.argv[2]
        
    # load neural network
    model = SRCNN.SRCNN()
    model.load_state_dict(torch.load(model_file))
    model.eval()

    image1 = get_weight_image(model.conv1.weight)    
    w = image1.width
    h = image1.height
    resize = torchvision.transforms.Resize((h * 8, w * 8), interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST)
    image1 = resize(image1)
    
    image1.save(image_file)
