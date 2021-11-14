Super-Resolution using Convolutional Neural Networks


This is a my repository, where I do my private experiments with Super Resolution using Convolutional Neural Networks (CNNs) for learning purposes. This work is inspired by paper "Image Super-Resolution Using Deep Convolutional Networks" by Dong et al. There are much better methods for super-resolution, based on GAN networks, however this paper is a good starting point getting some hands-on experience with CNNs.

The prerequisites to run the upsample.py script are python 3.8, pytorch 1.9.0+cu102, and PIL.

Some examples:

Original image upsampled with linear interpolation:
![linear interpolation](https://github.com/romanowicz/SuperRes/blob/master/output/waterfall_lr_linear.jpg?raw=true)

Super-resolved version:
![linear interpolation](https://github.com/romanowicz/SuperRes/blob/master/output/waterfall_lr_srcnn_3.jpg?raw=true)

Other examples can be found in the repository, in the 'output' subdirectory.

-Roman


