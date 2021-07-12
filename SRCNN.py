"""
Created on Mon Jul 12 14:02:10 2021

@author: rpa
"""

# Super Resolution CNN network from
# "Image Super-Resolution Using Deep Convolutional Networks"


import torch

        
class SRCNNParam():
    def __init__(self):
        self.c = 3
        self.f1 = 9
        self.f2 = 1
        self.f3 = 5
        self.n1 = 64
        self.n2 = 32
        
    
class SRCNN(torch.nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        p = SRCNNParam()
        self.conv1 = torch.nn.Conv2d(p.c, p.n1, kernel_size=p.f1, stride=1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(p.n1, p.n2, kernel_size=p.f2, stride=1)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(p.n2, p.c, kernel_size=p.f3, stride=1)
        self.relu3 = torch.nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.conv3(y)        
        y = self.relu3(y)
        return y
