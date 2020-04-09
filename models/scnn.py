# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 09:16:25 2020
Thanks to <<Spatial As Deep: Spatial CNN for Trafﬁc Scene Understanding>>
@author: 15652
"""


import torch as t 
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn import init
    

class SCNN(nn.Module):
    def __init__(self, C, w=1, h=1, bias=True):
        super(SCNN, self).__init__()
        assert w%2 == 1 and h%2 == 1, "w, h in SCNN should be odd" 
        self.h, self.w = h, w
        self.kernel_td = Parameter(t.zeros(C, 1, self.w, C))
        self.bias_td = Parameter(t.zeros(C)) if bias else None
        init.kaiming_normal_(self.kernel_td, mode='fan_out', nonlinearity='relu')
        self.kernel_dt = Parameter(t.zeros(C, 1, self.w, C))
        self.bias_dt = Parameter(t.zeros(C)) if bias else None
        init.kaiming_normal_(self.kernel_dt, mode='fan_out', nonlinearity='relu')
        self.kernel_lr = Parameter(t.zeros(C, 1, self.h, C))
        self.bias_lr = Parameter(t.zeros(C)) if bias else None
        init.kaiming_normal_(self.kernel_lr, mode='fan_out', nonlinearity='relu')
        self.kernel_rl = Parameter(t.zeros(C, 1, self.h, C))
        self.bias_rl = Parameter(t.zeros(C)) if bias else None
        init.kaiming_normal_(self.kernel_rl, mode='fan_out', nonlinearity='relu')
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()  # shape: N, H, W, C 
        H, W = x.size(1), x.size(2)
        p_td = ((self.w - 1)//2, 0)
        p_lr = ((self.h - 1)//2, 0)
        
        # top to down
        for H_ in range(1, H):
            temp = F.conv2d(x[:, H_-1].unsqueeze(1), self.kernel_td, padding=p_td).transpose(1, 3).contiguous()
            x[:, H_] += self.relu(temp.squeeze(1))
        # down to top
        for H_ in range(H-2, -1, -1):
            temp = F.conv2d(x[:, H_+1].unsqueeze(1), self.kernel_dt, padding=p_td).transpose(1, 3).contiguous()
            x[:, H_] += self.relu(temp.squeeze(1))
            
        # left to right
        temp = t.zeros_like(x)
        for W_ in range(1, W):
            temp = F.conv2d(x[:, :, W_-1].unsqueeze(1), self.kernel_lr, padding=p_lr).permute(0, 2, 3, 1).contiguous()
            x[:, :, W_] += self.relu(temp.squeeze(2))
        # right to left
        temp = t.zeros_like(x)
        for W_ in range(W-2, -1, -1):
            temp = F.conv2d(x[:, :, W_+1].unsqueeze(1), self.kernel_rl, padding=p_lr).permute(0, 2, 3, 1).contiguous()
            x[:, :, W_] += self.relu(temp.squeeze(2))
            
        x = x.permute(0, 3, 1, 2).contiguous()  # shape: N, C, H, W
        return x
            
            
if __name__ == "__main__":
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    inputs = t.randn(3, 64, 72, 128).to(device)
    print("inputs.shape:", inputs.size())
    model = SCNN(64, 9, 9).to(device)
    outputs = model(inputs)
    print("outputs.shape: ", outputs.size())
    print(outputs.max())
    print(outputs.min())