#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 14:53:58 2021

@author: rpa
"""
from PIL import Image

import torch
import torchvision
from torchvision.io import read_image, write_jpeg

import SRCNN
from SRCNNDataset import IMG_WIDTH
from SRCNNDataset import UPSCALING_FACTOR


model = SRCNN.SRCNN()
model.load_state_dict(torch.load("srcnn.pt"))
model.eval()

image = Image.open("train10000.jpg");

w = image.size[0]
h = image.size[1]

my_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((h * UPSCALING_FACTOR, w * UPSCALING_FACTOR))
])

image1 = my_transform(image)
#print(image1.size())

# TODO: convert to tiles tensor of size torch.Size([64, 3, 33, 33])

image2 = model(image1.unsqueeze(0))

m = torchvision.transforms.ToPILImage()
image3 = m(image2[0])

image3.save("output10000.png","PNG")

