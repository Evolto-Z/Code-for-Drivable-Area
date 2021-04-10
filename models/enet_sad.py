# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:39:18 2020

@author: 15652
"""


from base_modules import Flatten

import torch as t
from torch import nn
from torch.nn import functional as F
from base_modules import Conv_BN_ReLU


class Bottleneck(nn.Module):
    def __init__(self, in_chs, out_chs, dilation=1, prob=0, flag=None, conv_type=None):
        super(Bottleneck, self).__init__()
        self.flag = flag
        temp_chs = in_chs//2
        self.branch = nn.Sequential()
        if self.flag == 'up':
            self.residual = nn.ModuleList([
                nn.Conv2d(in_chs, out_chs, 1, 1, 0, bias=False),
                nn.MaxUnpool2d(2)
                ])
            self.branch.add_module('projection', Conv_BN_ReLU(in_chs, temp_chs, 1, 1, 0, relu_type='prelu', bias=False))
        elif self.flag == 'down':
            self.residual = nn.MaxPool2d(2, return_indices=True)
            self.branch.add_module('projection', Conv_BN_ReLU(in_chs, temp_chs, 2, 2, 0, relu_type='prelu', bias=False))
        elif self.flag == None:
            self.branch.add_module('projection', Conv_BN_ReLU(in_chs, temp_chs, 1, 1, 0, relu_type='prelu', bias=False))
        if conv_type == None:
            padding = dilation
            self.branch.add_module('conv', Conv_BN_ReLU(temp_chs, temp_chs, 3, 1, padding, dilation=dilation, relu_type='prelu'))
        elif conv_type == 'deconv':
            self.branch.add_module('conv', Conv_BN_ReLU(temp_chs, temp_chs, 3, 2, 1, conv_type='deconv', output_padding=1, relu_type='prelu'))
        elif conv_type == 'asymmetric':
            self.branch.add_module('conv', Conv_BN_ReLU(temp_chs, temp_chs, 5, 1, 2, conv_type='asymmetric', relu_type='prelu'))
        self.branch.add_module('expansion', Conv_BN_ReLU(temp_chs, out_chs, 1, 1, 0, relu_type=None))
        self.branch.add_module('regularizer', nn.Dropout2d(prob))
        self.prelu = nn.PReLU()
        
    def forward(self, x, max_indices=None):
        out = self.branch(x)
        if self.flag == 'up':
            out_res = self.residual[0](x)
            out_res = self.residual[1](out_res, max_indices)
        elif self.flag == 'down':
            out_res, max_indices = self.residual(x)
            # 进行通道padding
            n, c, h, w = out_res.size()
            c_pad = out.size(1) - c
            pad = t.zeros(n, c_pad, h, w).to(out_res.device)
            out_res = t.cat((out_res, pad), dim=1)
        elif self.flag == None:
            out_res = x
        out += out_res
        out = self.prelu(out)
        if self.flag == 'down':
            return out, max_indices
        return out


class ENet(nn.Module):
    def __init__(self, C, SAD=True, use_head=True):  # C是类别数, SAD是自注意力蒸馏
        super(ENet, self).__init__()
        self.SAD = SAD
        self.use_head = use_head
        self.C = C
        
        self.initial = nn.ModuleList([
            nn.Conv2d(3, 13, 3, 2, 1),
            nn.MaxPool2d(2)
            ])
        
        # encoder
        self.down1 = Bottleneck(16, 64, prob=0.01, flag='down')
        self.stage1 = nn.Sequential(
            Bottleneck(64, 64, prob=0.01),
            Bottleneck(64, 64, prob=0.01),
            Bottleneck(64, 64, prob=0.01),
            Bottleneck(64, 64, prob=0.01)
            )
        self.down2 = Bottleneck(64, 128, prob=0.1, flag='down')
        self.stage2 = nn.Sequential(
            Bottleneck(128, 128, prob=0.1),
            Bottleneck(128, 128, prob=0.1, dilation=2),
            Bottleneck(128, 128, prob=0.1, conv_type='asymmetric'),
            Bottleneck(128, 128, prob=0.1, dilation=4),
            Bottleneck(128, 128, prob=0.1),
            Bottleneck(128, 128, prob=0.1, dilation=8),
            Bottleneck(128, 128, prob=0.1, conv_type='asymmetric'),
            Bottleneck(128, 128, prob=0.1, dilation=16)
            )
        self.stage3 = nn.Sequential(
            Bottleneck(128, 128, prob=0.1),
            Bottleneck(128, 128, prob=0.1, dilation=2),
            Bottleneck(128, 128, prob=0.1, conv_type='asymmetric'),
            Bottleneck(128, 128, prob=0.1, dilation=4),
            Bottleneck(128, 128, prob=0.1),
            Bottleneck(128, 128, prob=0.1, dilation=8),
            Bottleneck(128, 128, prob=0.1, conv_type='asymmetric'),
            Bottleneck(128, 128, prob=0.1, dilation=16)
            )
        
        # decoder
        self.up2 = Bottleneck(128, 64, prob=0.1, flag='up', conv_type='deconv')
        self.stage4 = nn.Sequential(
            Bottleneck(64, 64, prob=0.1),
            Bottleneck(64, 64, prob=0.1)
            )
        self.up1 = Bottleneck(64, 16, prob=0.1, flag='up', conv_type='deconv')
        self.stage5 = nn.Sequential(
            Bottleneck(16, 16, prob=0.1)
            )
        self.fullconv = nn.ConvTranspose2d(16, C, 3, 2, 1, output_padding=1)
        
        # head: existence
        if self.use_head:
            self.head = nn.Sequential(
                nn.AvgPool2d(2),
                Flatten(),
                nn.Linear(128*32*32, 128),  # 这里的32*32是标准ENet的特征图尺寸
                nn.ReLU6(inplace=True),
                nn.Linear(128, C)
                )
    
    def at_gen(self, feature, scale_factor=1):
        attention = t.sum(t.pow(feature, 2), dim=1, keepdim=True)
        if scale_factor != 1:
            attention = F.interpolate(attention, scale_factor=scale_factor)
        attention = t.sigmoid(attention)
        return attention

    def forward(self, x):
        # encode
        out_1, out_2 = self.initial[0](x), self.initial[1](x)
        out = t.cat((out_1, out_2), dim=1)
        if self.SAD:
            attention = [self.at_gen(out, 0.5)]
        out, max_indices1 = self.down1(out)
        out = self.stage1(out)
        if self.SAD:
            attention.append(self.at_gen(out, 0.5))
        out, max_indices2 = self.down2(out)
        out = self.stage2(out)
        if self.SAD:
            attention.append(self.at_gen(out))
        out = self.stage3(out)
        if self.SAD:
            attention.append(self.at_gen(out))
            
        # head: existence
        existence = self.head(out)
        existence = t.softmax(existence, dim=1)
        
        # decode
        out = self.up2(out, max_indices2)
        out = self.stage4(out)
        out = self.up1(out, max_indices1)
        out = self.stage5(out)
        out = self.fullconv(out)
        out = t.softmax(out, dim=1)
        
        result = dict(out=out)
        if self.SAD:
            result['attention'] = attention
        if self.use_head:
            result['existence'] = existence
        return result
    
    
if __name__ == "__main__":
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    inputs = t.randn(2, 3, 512, 512).to(device)
    print("inputs.shape:", inputs.size())
    model = ENet(5).to(device)
    result = model(inputs)
    print("out.shape: ", result['out'].size())
    print(result['out'].max())
    print(result['out'].min())
    print("attention[3].shape: ", result['attention'][3].size())
    print(result['attention'][3].max())
    print(result['attention'][3].min())
    print("existence: ", result['existence']) 