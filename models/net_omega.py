# -*- coding: utf-8 -*-
"""
Created on Tue May  5 13:10:58 2020

@author: 15652
"""


# 自建库
from .model_wrapper import WrappedModule
from .hourglass_omega import Hourglass, Bottleneck
from .base_modules import Conv_BN_ReLU

import torch as t
from torch import nn


class NetOmega(WrappedModule):
    def __init__(self, sad=True):
        super(NetOmega, self).__init__()
        self.model_name = 'Net_Delta'
        self.sad = sad
        
        self.initial = nn.ModuleList([
            nn.Conv2d(4, 12, 3, 2, 1),
            nn.MaxPool2d(2)
            ])
        self.blk1 = nn.Sequential(
            Bottleneck(16, 64, 'down'),
            Bottleneck(64, 64),
            Bottleneck(64, 64),
            Bottleneck(64, 128, 'down'),
            Bottleneck(128, 128),
            Bottleneck(128, 128),
            )
        self.blk2_1 = nn.Sequential(
            Bottleneck(128, 128),
            Bottleneck(128, 128),
            Bottleneck(128, 64),
            Bottleneck(64, 64),
            Bottleneck(64, 64)
            )
        self.blk2_2 = nn.Sequential(
            Bottleneck(128, 128),
            Bottleneck(128, 128),
            Bottleneck(128, 64),
            Bottleneck(64, 64),
            Bottleneck(64, 64)
            )
        
        self.res1 = nn.Conv2d(128, 128, 1, 1, 0, bias=False)
        self.res2 = nn.Conv2d(64, 128, 1, 1, 0, bias=False)
        
        self.confidence = nn.Sequential(
            Bottleneck(64, 64),
            Bottleneck(64, 32),
            Bottleneck(32, 32),
            Bottleneck(32, 32),
            Bottleneck(32, 16),
            Bottleneck(16, 16),
            nn.Conv2d(16, 6, 1, 1, 0)
            )
        self.offset = nn.Sequential(
            Bottleneck(64, 64),
            Bottleneck(64, 32),
            Bottleneck(32, 32),
            Bottleneck(32, 32),
            nn.Conv2d(32, 2, 1, 1, 0)
            )
        
        self.stage1 = Hourglass(128, output_padding=[1, 1, 1, [0, 0]])
        self.stage2 = Hourglass(128, output_padding=[1, 1, 1, [0, 0]])
    
    def at_gen(self, feature):
        attention = t.sum(t.pow(feature, 2), dim=1, keepdim=True)
        attention = t.sigmoid(attention)
        return attention
    
    def forward(self, x):
        x_1, x_2 = self.initial[0](x), self.initial[1](x)
        x = t.cat([x_1, x_2], dim=1)
        x = self.blk1(x)
        out1 = self.blk2_1(self.stage1(x))
        if self.sad:
            attention = [self.at_gen(out1)]
        out1 = self.res1(x) + self.res2(out1)
        out2 = self.blk2_2(self.stage2(out1))
        if self.sad:
            attention.append(self.at_gen(out2))
        confidence = self.confidence(out2)
        offset = self.offset(out2)
        
        confidence = t.softmax(confidence, dim=1)
        offset = t.sigmoid(offset)
        
        result = dict(confidence=confidence, offset=offset)
        
        return result


if __name__ == "__main__":
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    inputs = t.rand(2, 3, 144, 256).to(device)
    print("inputs.shape:", inputs.size())
    model = NetDelta().to(device)
    outputs = model(inputs)
    print("confidence.shape: ", outputs[1]['confidence'].size())
    print("offset.shape: ", outputs[1]['offset'].size())
    print("existence.shape: ", outputs[1]['existence'].size())