# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:16:13 2020

@author: 15652
"""

# 自建库
from .base_modules import Conv_BN_ReLU

from torch import nn
import torch as t


class Bottleneck(nn.Module):
    def __init__(self, in_chs, out_chs, flag=None, output_padding=0, dilation=1, relu_type='prelu', drop_prob=0):
        super(Bottleneck, self).__init__()
        self.flag = flag
        temp_chs = in_chs//2
        self.conv1 = Conv_BN_ReLU(in_chs, temp_chs, 1, 1, 0, relu_type=relu_type, bias=False)
        if self.flag == None:
            padding = dilation
            self.conv2 = Conv_BN_ReLU(temp_chs, temp_chs, 3, 1, padding, dilation=dilation, relu_type=relu_type)
            self.residual = Conv_BN_ReLU(in_chs, out_chs, 1, 1, 0, relu_type=None, bias=False)
        elif self.flag == 'down':
            self.conv2 = Conv_BN_ReLU(temp_chs, temp_chs, 3, 2, 1, relu_type=relu_type)
            self.residual = Conv_BN_ReLU(in_chs, out_chs, 3, 2, 1, relu_type=relu_type)
        elif self.flag == 'up':
            self.conv2 = Conv_BN_ReLU(temp_chs, temp_chs, 3, 2, 1, output_padding=output_padding, conv_type='deconv', relu_type=relu_type)
            self.residual = Conv_BN_ReLU(in_chs, out_chs, 3, 2, 1, output_padding=output_padding, conv_type='deconv', relu_type=relu_type)
        else:
            raise AssertionError("flag should be 'up', 'down' or None")
        self.conv3 = Conv_BN_ReLU(temp_chs, out_chs, 1, 1, 0, relu_type=None)
        self.regularizer = nn.Dropout2d(drop_prob)
        self.prelu = nn.PReLU()
        
    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.regularizer(x)
        res = self.residual(res)
        x += res
        x = self.prelu(x)
        return x 


# 4阶Hourglass，融合多尺度特征
class Hourglass(nn.Module):
    def __init__(self, in_chs, output_padding=[1, 1, 1, 1], relu_type='prelu', drop_prob=0):
        super(Hourglass, self).__init__()
        out_chs = in_chs
        self.down1 = Bottleneck(in_chs, out_chs, 'down', relu_type=relu_type, drop_prob=drop_prob)
        self.down2 = Bottleneck(out_chs, out_chs, 'down', relu_type=relu_type, drop_prob=drop_prob)
        self.down3 = Bottleneck(out_chs, out_chs, 'down', relu_type=relu_type, drop_prob=drop_prob)
        self.down4 = Bottleneck(out_chs, out_chs, 'down', relu_type=relu_type, drop_prob=drop_prob)
        self.same = nn.Sequential(
                Bottleneck(out_chs, out_chs, relu_type=relu_type, drop_prob=drop_prob),
                Bottleneck(out_chs, out_chs, relu_type=relu_type, drop_prob=drop_prob),
                Bottleneck(out_chs, out_chs, relu_type=relu_type, drop_prob=drop_prob)
            )
        self.up1 = Bottleneck(out_chs, out_chs, 'up', output_padding[0], relu_type=relu_type, drop_prob=drop_prob)
        self.up2 = Bottleneck(out_chs, out_chs, 'up', output_padding[1], relu_type=relu_type, drop_prob=drop_prob)
        self.up3 = Bottleneck(out_chs, out_chs, 'up', output_padding[2], relu_type=relu_type, drop_prob=drop_prob)
        self.up4 = Bottleneck(out_chs, out_chs, 'up', output_padding[3], relu_type=relu_type, drop_prob=drop_prob)
        self.residual1 = Bottleneck(in_chs, out_chs, relu_type=relu_type, drop_prob=drop_prob)
        self.residual2 = Bottleneck(out_chs, out_chs, relu_type=relu_type, drop_prob=drop_prob)
        self.residual3 = Bottleneck(out_chs, out_chs, relu_type=relu_type, drop_prob=drop_prob)
        self.residual4 = Bottleneck(out_chs, out_chs, relu_type=relu_type, drop_prob=drop_prob)
        
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