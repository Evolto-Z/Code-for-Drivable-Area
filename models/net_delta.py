# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 22:11:03 2020

@author: 15652
"""


# 自建库
from .model_wrapper import WrappedModule
from .hourglass import Hourglass, Bottleneck

import torch as t
from torch import nn


class NetDelta(WrappedModule):
    def __init__(self, use_head=True):
        super(NetDelta, self).__init__()
        self.model_name = 'Net_Delta'
        self.use_head = use_head
        
        self.initial = nn.ModuleList([
            nn.Conv2d(3, 13, 3, 2, 1),
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
            Bottleneck(128, 64),
            Bottleneck(64, 64)
            )
        self.blk2_2 = nn.Sequential(
            Bottleneck(128, 64),
            Bottleneck(64, 64)
            )
        
        self.confidence_res = nn.Conv2d(6, 128, 1, 1, 0, bias=False)
        self.out_res = nn.Conv2d(64, 128, 1, 1, 0, bias=False)
        
        self.confidence1 = nn.Sequential(
            Bottleneck(64, 32),
            Bottleneck(32, 32),
            nn.Conv2d(32, 6, 1, 1, 0)
            )
        self.offset1 = nn.Sequential(
            Bottleneck(64, 32),
            nn.Conv2d(32, 2, 1, 1, 0)
            )
        self.confidence2 = nn.Sequential(
            Bottleneck(64, 32),
            Bottleneck(32, 32),
            nn.Conv2d(32, 6, 1, 1, 0)
            )
        self.offset2 = nn.Sequential(
            Bottleneck(64, 32),
            nn.Conv2d(32, 2, 1, 1, 0)
            )
        
        self.stage1 = Hourglass(128, output_padding=[1, 1, 1, [0, 1]])
        self.stage2 = Hourglass(128, output_padding=[1, 1, 1, [0, 1]])
    
    def forward(self, x):
        x_1, x_2 = self.initial[0](x), self.initial[1](x)
        x = t.cat((x_1, x_2), dim=1)
        
        x = self.blk1(x)
        out1 = self.blk2_1(self.stage1(x))
        confidence1 = self.confidence1(out1)
        offset1 = self.offset1(out1)
        out1 = x + self.out_res(out1) + self.confidence_res(confidence1)
        out2 = self.blk2_2(self.stage2(out1))
        confidence2 = self.confidence2(out2)
        offset2 = self.offset2(out2)
        
        confidence1 = t.softmax(confidence1, dim=1)
        confidence2 = t.softmax(confidence2, dim=1)
        offset1 = t.sigmoid(offset1)
        offset2 = t.sigmoid(offset2)
        
        result1 = dict(confidence=confidence1, offset=offset1)
        result2 = dict(confidence=confidence2, offset=offset2)
        return result1, result2


if __name__ == "__main__":
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    inputs = t.rand(2, 3, 576, 1024).to(device)
    print("inputs.shape:", inputs.size())
    model = NetDelta().to(device)
    outputs = model(inputs)
    print("confidence.shape: ", outputs[1]['confidence'].size())
    print("offset.shape: ", outputs[1]['offset'].size())
    print("parameters(M): ", model.count_parameters()/10e6)