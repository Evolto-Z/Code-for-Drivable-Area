# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:01:33 2020

@author: 15652
"""


import torch as t
from torch import nn
from torch.nn import functional as F


# 将卷积、BatchNorm2d、ReLU封装在一起，可通过kw设置bias和dilation
class Conv_BN_ReLU(nn.Module):
    def __init__(self, in_chs, out_chs, k, s=1, p=0, conv_type=None, relu_type='relu', inplace=False, **kw):
        super(Conv_BN_ReLU, self).__init__()
        if conv_type == 'deconv':
            self.unit = nn.Sequential(
                nn.ConvTranspose2d(in_chs, out_chs, k, s, p, **kw)
                )
        elif conv_type == 'asymmetric':
            assert in_chs == out_chs, "in_chs and out_chs in asymmetric should be the same"
            self.unit = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, (k, 1), (s, 1), (p, 0), **kw),
                nn.Conv2d(in_chs, in_chs, (1, k), (1, s), (0, p), **kw)
                )
        elif conv_type == None:
            self.unit = nn.Sequential(
                nn.Conv2d(in_chs, out_chs, k, s, p, **kw)
                )
        if relu_type != None:
            self.unit.add_module('BN', nn.BatchNorm2d(out_chs))
            if relu_type == 'relu':
                self.unit.add_module('ReLU', nn.ReLU(inplace=inplace))
            elif relu_type == 'prelu':
                self.unit.add_module('PReLU', nn.PReLU())
            elif relu_type == 'relu6':
                self.unit.add_module('ReLU6', nn.ReLU6(inplace=inplace))
            
    def forward(self, x):
        x = self.unit(x)
        return x

    
# 将卷积、BatchNorm2d、ReLU封装在一起，可通过kw设置bias和dilation
class Conv_ReLU(nn.Module):
    def __init__(self, in_chs, out_chs, k, s=1, p=0, conv_type=None, relu_type='relu', inplace=False, **kw):
        super(Conv_ReLU, self).__init__()
        if conv_type == 'deconv':
            self.unit = nn.Sequential(
                nn.ConvTranspose2d(in_chs, out_chs, k, s, p, **kw)
                )
        elif conv_type == 'asymmetric':
            assert in_chs == out_chs, "in_chs and out_chs in asymmetric should be the same"
            self.unit = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, (k, 1), (s, 1), (p, 0), **kw),
                nn.Conv2d(in_chs, in_chs, (1, k), (1, s), (0, p), **kw)
                )
        elif conv_type == None:
            self.unit = nn.Sequential(
                nn.Conv2d(in_chs, out_chs, k, s, p, **kw)
                )
        if relu_type != None:
            if relu_type == 'relu':
                self.unit.add_module('ReLU', nn.ReLU(inplace=inplace))
            elif relu_type == 'prelu':
                self.unit.add_module('PReLU', nn.PReLU())
            elif relu_type == 'relu6':
                self.unit.add_module('ReLU6', nn.ReLU6(inplace=inplace))
            
    def forward(self, x):
        x = self.unit(x)
        return x


# 将张量展开
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        return x.view(x.size(0), -1)
    
    
# Spatial Attention Module，空间域的注意力机制
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, 1, 3)
    
    def forward(self, x):
        max_map, _ = x.max(dim=1)
        avg_map = x.mean(dim=1)
        attention = t.stack([avg_map, max_map], dim=1)
        attention = F.relu(self.conv(attention))
        x = x*attention
        return x

    
# Squeeze and Excitation Block，通道域的注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_chs, ratio=16):
        super(ChannelAttention, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        temp_chs = in_chs//ratio
        out_chs = in_chs
        self.excitation = nn.Sequential(
            nn.Conv2d(in_chs, temp_chs, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(temp_chs, out_chs, 1, 1, 0)
            )
        
    def forward(self, x):
        attention = self.excitation(self.squeeze(x))
        attention = t.sigmoid(attention)
        x = x*attention
        return x
    
    
# Convolutional Block Attention Module，混合域的注意力机制
class CBAM(nn.Module):
    def __init__(self, in_chs, ratio=16):
        super(CBAM, self).__init__()
        temp_chs = in_chs//ratio
        out_chs = in_chs
        self.fc = nn.Sequential(
            nn.Conv2d(in_chs, temp_chs, 1, 1, 0),
            nn.Conv2d(temp_chs, out_chs, 1, 1, 0)
            )
        self.squeeze_max = nn.AdaptiveMaxPool2d((1, 1))
        self.squeeze_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.spatial = SpatialAttention()
        
    def forward(self, x):
        max_attention = self.squeeze_max(x)
        avg_attention = self.squeeze_avg(x)
        max_attention = self.fc(max_attention)
        avg_attention = self.fc(avg_attention)
        ch_attention = t.sigmoid(max_attention + avg_attention)
        x = x*ch_attention
        x = self.spatial(x)
        return x
    

# 测试最为复杂的CBAM
if __name__=='__main__':
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    inputs = t.randn(3, 99, 6, 8).to(device)
    print('inputs.shape: ', inputs.size())
    model = CBAM(99).to(device)
    outputs = model(inputs)
    print('outputs.shape: ', outputs.size())