# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:10:14 2020
Thanks to <<HarDNet: A Low Memory Trafﬁc Network>>
@author: 15652
"""

# 自建库
from .base_modules import Conv_BN_ReLU, Flatten

import torch as t
import torch.nn as nn
import math


class HarDBlock(nn.Module):
    def __init__(self, chs, growth_rate, compress_factor, n_layers, dw=False):
        super().__init__()
        self.block = nn.ModuleList([])
        self.link_memo, out_chs_memo = self.get_memo(chs, growth_rate, compress_factor, n_layers)
        for n, link in enumerate(self.link_memo):
            if link == []:
                continue
            in_chs = 0
            for i in link:
                in_chs += out_chs_memo[i]
            if dw:
                self.block.append(nn.Sequential(
                    Conv_BN_ReLU(in_chs, out_chs_memo[n], 3, 1, 1, bias=False),
                    Conv_BN_ReLU(out_chs_memo[n], out_chs_memo[n], 3, 1, 1, relu=False, groups=out_chs_memo[n], bias=False)
                    ))
            else:
                self.block.append(Conv_BN_ReLU(in_chs, out_chs_memo[n], 3, 1, 1, bias=False))
        self.out_chs = 0
        self.final_link = []
        for i, out_chs in enumerate(out_chs_memo):
            if i == 0 or i%2 == 1 or i == n_layers:
                self.final_link.append(i)
                self.out_chs += out_chs
        
    def get_memo(self, chs, growth_rate, compress_factor, n_layers):
        link_memo = [[]]
        out_chs_memo = [chs]
        n_max = int(math.log2(n_layers)) 
        for n in range(1, n_layers+1):
            link = []
            for i in reversed(range(n_max+1)):
                dv = 2**i
                if n%dv == 0:
                    k = n - dv
                    link.append(k)
            out_chs_memo.append(int(growth_rate*compress_factor**(len(link) - 1)))
            link_memo.append(link)
        return link_memo, out_chs_memo
        
    def forward(self, x):
        out = [x, ]
        for i, layer in enumerate(self.block, start=1):
            link = self.link_memo[i]
            temp = [out[i] for i in link]
            out.append(layer(t.cat(temp, dim=1)))
        out = [out[i] for i in self.final_link]
        out = t.cat(out, dim=1)
        return out
        
    
class HarDNet(nn.Module):
    def __init__(self, dw=False, use_maxpool=False, is_output=True):
        super().__init__()
        #HarDNet68
        second_kernel = 3
        max_pool = True
        compress_factor = 1.7
        drop_rate = 0.1
        init_chs =   [32, 64]
        chs_list =    [  128, 256, 320, 640, 1024]
        growth_rate = [   14,  16,  20,  40,  160]
        n_layers =    [    8,  16,  16,  16,    4]
        down_sample = [    1,   0,   1,   1,    0]          
        if dw:
            second_kernel = 1
            max_pool = False
            drop_rate = 0.05
            
        self.model = nn.ModuleList([])
        
        self.model.append(Conv_BN_ReLU(3, init_chs[0], 3, 2, 1, bias=False))
        self.model.append(Conv_BN_ReLU(init_chs[0], init_chs[1], second_kernel, bias=False))
        if max_pool:
            self.model.append(nn.MaxPool2d(3, 2, 1))
        else:
            self.model.append(Conv_BN_ReLU(init_chs[1], init_chs[1], 3, 2, 1, relu=False, bias=False))
        chs = init_chs[1]
        for i in range(len(chs_list)):
            blk = HarDBlock(chs, growth_rate[i], compress_factor, n_layers[i], dw=dw)
            self.model.append(blk)
            self.model.append(Conv_BN_ReLU(blk.out_chs, chs_list[i], 1, 1, 0, bias=False))
            chs = chs_list[i]
            if down_sample[i] == 1:
                self.model.append(nn.MaxPool2d(2, 2))
            else:
                self.model.append(Conv_BN_ReLU(chs, chs, 3, 2, 1, relu=False, bias=False))   
        
        if is_output:
            chs = chs_list[-1]
            self.model.append (
                nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    Flatten(),
                    nn.Dropout(drop_rate),
                    nn.Linear(chs, 1000) 
                    ))
          
    def forward(self, x):
        for i, layer in enumerate(self.model):
            x = layer(x)
        return x
    
    
if __name__ == '__main__':
    model = HarDNet(dw=True)
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    model.to(device)
    inputs = t.rand(2, 3, 512, 512).to(device)
    outputs = model(inputs)
    print("outputs.shape: ", outputs.size())