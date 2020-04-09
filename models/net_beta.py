# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 11:43:38 2020

@author: 15652
"""


# 自建库
from .model_wrapper import WrappedModule
from .hourglass import Hourglass
from .base_modules import Bottleneck, Conv_BN_ReLU

import torch as t
from torch import nn

class NetBeta(WrappedModule):
    def __init__(self):
        super(NetBeta, self).__init__()
        self.model_name = 'Net_Beta'
        self.preprocess = nn.Sequential(
            Conv_BN_ReLU(3, 16, 7, 2, 3),
            Bottleneck(16, 32),
            nn.MaxPool2d(2),
            Bottleneck(32, 32),
            Bottleneck(32, 64)
            )
        self.postprocess = nn.Sequential(
            Bottleneck(64, 64),
            Conv_BN_ReLU(64, 64, 1, 1, 0),
            )
        
        self.confidence_res = nn.Conv2d(6, 64, 1, 1, 0)
        self.out_res = nn.Conv2d(64, 64, 1, 1, 0)
        
        self.confidence = nn.Conv2d(64, 6, 1)
        self.offset = nn.Conv2d(64, 2, 1)
        
        self.stage1 = Hourglass(64, (1, 1, 1, (0, 1)))
        self.stage2 = Hourglass(64, (1, 1, 1, (0, 1)))
    
    def forward(self, x):
        x = self.preprocess(x)
        out1 = self.postprocess(self.stage1(x))
        confidence1 = self.confidence(out1)
        offset1 = self.offset(out1)
        out1 = x + self.out_res(out1) + self.confidence_res(confidence1)
        out2 = self.postprocess(self.stage2(out1))
        confidence2 = self.confidence(out2)
        offset2 = self.offset(out2)
        confidence1 = t.softmax(confidence1, dim=1)
        confidence2 = t.softmax(confidence2, dim=1)
        return (confidence1, offset1), (confidence2, offset2)
    
    
if __name__ == "__main__":
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    inputs = t.rand(2, 3, 144, 256).to(device)
    print("inputs.shape:", inputs.size())
    model = NetBeta().to(device)
    outputs = model(inputs)
    print("confidence.shape: ", outputs[1][0].size())
    print("offset.shape: ", outputs[1][1].size())