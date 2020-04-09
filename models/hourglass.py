# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:16:13 2020

@author: 15652
"""

# 自建库
from .base_modules import Bottleneck

from torch import nn
import torch as t


# 4阶Hourglass，融合多尺度特征，output_padding需要自行计算
class Hourglass(nn.Module):
    def __init__(self, in_chs, output_padding):
        super(Hourglass, self).__init__()
        out_chs = in_chs
        self.down1 = Bottleneck(in_chs, out_chs, 'down')
        self.down2 = Bottleneck(out_chs, out_chs, 'down')
        self.down3 = Bottleneck(out_chs, out_chs, 'down')
        self.down4 = Bottleneck(out_chs, out_chs, 'down')
        self.same = nn.Sequential(
                Bottleneck(out_chs, out_chs),
                Bottleneck(out_chs, out_chs),
                Bottleneck(out_chs, out_chs)
            )
        self.up1 = Bottleneck(out_chs, out_chs, 'up', output_padding[0])
        self.up2 = Bottleneck(out_chs, out_chs, 'up', output_padding[1])
        self.up3 = Bottleneck(out_chs, out_chs, 'up', output_padding[2])
        self.up4 = Bottleneck(out_chs, out_chs, 'up', output_padding[3])
        self.residual1 = Bottleneck(in_chs, out_chs)
        self.residual2 = Bottleneck(out_chs, out_chs)
        self.residual3 = Bottleneck(out_chs, out_chs)
        self.residual4 = Bottleneck(out_chs, out_chs)
        
    def forward(self, x):
        res = x 
        out1 = self.down1(x)
        out2 = self.down2(out1)
        out3 = self.down3(out2)
        out4 = self.down4(out3)
        out = self.same(out4)
        out = self.up4(out) + self.residual4(out3)
        out = self.up3(out) + self.residual3(out2)
        out = self.up2(out) + self.residual2(out1)
        out = self.up1(out) + self.residual1(res)
        return out        
        
        
if __name__ == "__main__":
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    inputs = t.randn(2, 64, 144, 256).to(device)
    print("input.shape:", inputs.size())
    model = Hourglass(64, (1, 1, 1, 1)).to(device)
    outputs = model(inputs)
    print("outputs.shape: ", outputs.size())