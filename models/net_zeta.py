# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 12:23:24 2020

@author: 15652
"""


# 自建库
from .model_wrapper import WrappedModule
from .hourglass import Hourglass
from .base_modules import Bottleneck, Conv_BN_ReLU, CBAM
from .scnn import SCNN

import torch as t
from torch import nn

class NetZeta(WrappedModule):
    def __init__(self):
        super(NetZeta, self).__init__()
        self.model_name = 'Net_Zeta'
        self.preprocess = nn.Sequential(
            Conv_BN_ReLU(3, 16, 7, 2, 3),
            Bottleneck(16, 32),
            nn.MaxPool2d(2),
            Bottleneck(32, 32),
            Bottleneck(32, 64)
            )
        self.hourglass = Hourglass(64, (1, 1, 1, (0, 1)))
        self.transition = nn.Sequential(
            Bottleneck(64, 64),
            Conv_BN_ReLU(64, 64, 1, 1, 0)
            )
        self.out_res = nn.Conv2d(64, 64, 1, 1, 0)
        self.confidence1_res = nn.Conv2d(6, 64, 1, 1, 0)
        self.scnn = SCNN(64, 9, 9)
        
        self.confidence1 = nn.Conv2d(64, 6, 1, 1, 0)
        self.confidence2 = nn.Conv2d(64, 6, 1, 1, 0)
        self.offset = nn.Conv2d(64, 2, 1, 1, 0)
    
    def forward(self, x):
        x = self.preprocess(x)
        out = self.hourglass(x)
        out = self.transition(out)
        confidence1 = self.confidence1(out)
        offset = self.offset(out)
        x = x + self.out_res(out) + self.confidence1_res(confidence1)
        x = self.scnn(x)
        confidence2 = self.confidence2(out)
        confidence1 = t.softmax(confidence1, dim=1)
        confidence2 = t.softmax(confidence2, dim=1)
        return confidence1, confidence2, offset
    
    
if __name__ == "__main__":
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    inputs = t.rand(2, 3, 288, 512).to(device)
    print("inputs.shape:", inputs.size())
    model = NetGamma().to(device)
    outputs = model(inputs)
    print("confidence.shape: ", outputs[0].size())
    print("offset.shape: ", outputs[1].size())