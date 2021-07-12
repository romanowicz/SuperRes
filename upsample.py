#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 14:53:58 2021

@author: rpa
"""

import sys
import torch
import torchvision
from PIL import Image

import SRCNN
from SRCNNDataset import UPSCALING_FACTOR


def usage():
    print("usage: upsample.py <input_image> <output_image>")
    sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) != 3:    
        usage()

    input_image = sys.argv[1]
    output_image = sys.argv[2]
        
    model = SRCNN.SRCNN()
    model.load_state_dict(torch.load("srcnn.pt"))
    model.eval()
    
    image = Image.open(input_image);
    
    w = image.size[0]
    h = image.size[1]
    
    my_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((h * UPSCALING_FACTOR, w * UPSCALING_FACTOR), torchvision.transforms.InterpolationMode.BILINEAR)
    ])
    
    image1 = my_transform(image)
    image2 = model(image1.unsqueeze(0))    
    image2 = image2 / torch.max(image2)

    m = torchvision.transforms.ToPILImage()
    image3 = m(image2[0])
    image3.save(output_image)
        
