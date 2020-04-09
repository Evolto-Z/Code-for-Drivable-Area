# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:01:33 2020

@author: 15652
"""


import torch as t
from torch import nn
from torch.nn import functional as F


# 将卷积、BatchNorm2d、ReLU封装在一起，卷积包括了普通卷积、反卷积，其他卷积类型可通过kw调整卷积参数进行设置
class Conv_BN_ReLU(nn.Module):
    def __init__(self, in_chs, out_chs, k_size, s=1, p=0, deconv=False, relu=True, inplace=False, **kw):
        super(Conv_BN_ReLU, self).__init__()
        if deconv:
            self.unit = nn.Sequential(
                    nn.ConvTranspose2d(in_chs, out_chs, k_size, s, p, **kw)
                )
        else:
            self.unit = nn.Sequential(
                    nn.Conv2d(in_chs, out_chs, k_size, s, p, **kw)
                )
        self.unit.add_module('1', nn.BatchNorm2d(out_chs))
        if relu:
            self.unit.add_module('2', nn.ReLU(inplace=inplace))
            
    def forward(self, x):
        x = self.unit(x)
        return x


# 瓶颈层，将等采样、下采样、上采样三种形式封装在一起
class Bottleneck(nn.Module):
    def __init__(self, in_chs, out_chs, flag=None, output_padding=0):
        super(Bottleneck, self).__init__()
        self.flag = flag
        temp_chs = out_chs//2
        self.conv1 = Conv_BN_ReLU(in_chs, temp_chs, 1, 1, 0)
        if self.flag == None:
            self.conv2 = Conv_BN_ReLU(temp_chs, temp_chs, 3, 1, 1)
            self.residual = Conv_BN_ReLU(in_chs, out_chs, 1, 1, 0)
        elif self.flag == 'down':
            self.conv2 = Conv_BN_ReLU(temp_chs, temp_chs, 3, 2, 1)
            self.residual = Conv_BN_ReLU(in_chs, out_chs, 3, 2, 1)
        elif self.flag == 'up':
            self.conv2 = Conv_BN_ReLU(temp_chs, temp_chs, 3, 2, 1, output_padding=output_padding, deconv=True)
            self.residual = Conv_BN_ReLU(in_chs, out_chs, 3, 2, 1, output_padding=output_padding, deconv=True)
        else:
            raise AssertionError("flag should be 'up', 'down' or None")
        self.conv3 = Conv_BN_ReLU(temp_chs, out_chs, 1, 1, 0)
        
    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        res = self.residual(res)
        x += res
        return x 
    
    
# 将张量沿通道维展开
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