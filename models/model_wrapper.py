# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 11:52:29 2020

@author: 15652
"""


import torch as t
from torch import nn
import os
import time


# 对nn.Module进行一次封装
class WrappedModule(nn.Module):
    def __init__(self):
        super(WrappedModule, self).__init__()
        self.model_name = str(type(self))

    def count_parameters(self):
        return sum(params.numel() for params in self.parameters() if params.requires_grad)
        
    def save(self, name=None):
        if name is None:
            cronus = time.strftime('%m_%d_%H_%M', time.localtime())
            name = os.path.join('checkpoints/', cronus+'_'+self.model_name+'.pth')
        t.save(self.state_dict(), name)
        return name
    
    def load(self, root):
        self.load_state_dict(t.load(root))