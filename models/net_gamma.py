# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 22:41:07 2020

@author: 15652
"""


# 自建库
from .model_wrapper import WrappedModule
from .hourglass import Hourglass
from .base_modules import Bottleneck, Conv_BN_ReLU, CBAM
from .scnn import SCNN

import torch as t
from torch import nn

class NetGamma(WrappedModule):
    def __init__(self):
        super(NetGamma, self).__init__()
        self.model_name = 'Net_Gamma'
        self.preprocess = nn.Sequential(
            Conv_BN_ReLU(3, 16, 7, 2, 3),
            Bottleneck(16, 32),
            nn.MaxPool2d(2),
            Bottleneck(32, 32),
            Bottleneck(32, 64)
            )
        self.stage = nn.Sequential(
            Hourglass(64, (1, 1, 1, (0, 1))),
            SCNN(64, 3, 3)
            )
        self.attention_blk = CBAM(64, 8)
        self.postprocess = nn.Sequential(
            Bottleneck(64, 64),
            Conv_BN_ReLU(64, 64, 1, 1, 0)
            )
        
        self.confidence = nn.Conv2d(64, 6, 1, 1, 0)
        self.offset = nn.Conv2d(64, 2, 1, 1, 0)
    
    def forward(self, x):
        x = self.preprocess(x)
        x = self.stage(x)
        x = self.attention_blk(x)
        x = self.postprocess(x)
        confidence = self.confidence(x)
        offset = self.offset(x)
        confidence = t.softmax(confidence, dim=1)
        return confidence, offset
    
    
if __name__ == "__main__":
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    inputs = t.rand(2, 3, 288, 512).to(device)
    print("inputs.shape:", inputs.size())
    model = NetGamma().to(device)
    outputs = model(inputs)
    print("confidence.shape: ", outputs[0].size())
    print("offset.shape: ", outputs[1].size())