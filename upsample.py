#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 14:53:58 2021

@author: Roman Parys (romanowicz@protonmail.ch)
"""

import sys
import torch
import torchvision

import matplotlib.pyplot as plot
from PIL import Image

import SRCNN


UPSCALING_FACTOR = 3


def usage():
    print("usage: upsample.py <input_image> <output_image> [model.pt]")
    sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) != 3 and len(sys.argv) != 4:    
        usage()

    input_image = sys.argv[1]
    output_image = sys.argv[2]
    model_file = "srcnn.pt"
    if len(sys.argv) == 4:
        model_file = sys.argv[3]
        
    # load neural network
    model = SRCNN.SRCNN()
    model.load_state_dict(torch.load(model_file))
    model.eval()

    # open and resize image
    image = Image.open(input_image);
    w = image.size[0]
    h = image.size[1]
    resize = torchvision.transforms.Resize((h * UPSCALING_FACTOR, w * UPSCALING_FACTOR), interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST)
    image = resize(image)
    image_ycbcr = image.convert('YCbCr')
    image_y, image_cb, image_cr = image_ycbcr.split()
    
    # make tensor from the luminance channel and perform inference
    to_tensor = torchvision.transforms.ToTensor()
    image_t = to_tensor(image_y)
    image_t = image_t.unsqueeze(0)
    
    o_median = torch.median(image_t)
    o_min = torch.min(image_t)
    o_max = torch.max(image_t)
    
    image_t = model(image_t)
    
    t_min = torch.min(image_t)
    t_max = torch.max(image_t)
    t_median = torch.median(image_t)   
    
    # normalize brightness
    factor = o_median / t_median
    image_t = torch.clamp(image_t * factor, min=0.0, max=1.0)
    
    # assemble luminance and Cb, Cr channels
    to_image = torchvision.transforms.ToPILImage()
    image_y = to_image(image_t[0])
    crop = torchvision.transforms.CenterCrop(image_t.size()[2:])
    image_ycrcb = Image.merge('YCbCr',(image_y, crop(image_cb), crop(image_cr)))

    #convert to RGB and save
    image = image_ycrcb.convert('RGB')
    image.save(output_image)
