# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:14:33 2020

@author: 15652
"""


import torch as t
from torch import nn


class LossFunc(nn.Module):
    def __init__(self, opt):
        super(LossFunc, self).__init__()
        self.coeff = opt.coeff
        self.bg_weight = opt.bg_weight
            
    def confidence_loss(self, confidence, instance):
        confidence = confidence.gather(3, instance.long())
        loss = -t.log(confidence)
        loss[instance==0] *= self.bg_weight  # 背景的交叉熵权重要小一点
        loss = loss.mean()
        return loss
    
    def offset_loss(self, offset, gt_offset, instance):
        mask = instance.squeeze(-1).ne(0)
        temp = t.pow((gt_offset - offset)[mask], 2).sum(dim=-1)
        loss = 0.
        stt = 0
        for num_points in mask.sum(dim=(1, 2)):
            num_points = num_points.item()
            if num_points == 0:
                continue
            loss += temp[stt: stt+num_points].mean()
            stt += num_points
        loss = loss/mask.size(0)
        return loss
    
    def forward(self, predictions, targets):
        confidence, offset = predictions['confidence'], predictions['offset']
        instance, gt_offset = targets
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        offset = offset.permute(0, 2, 3, 1).contiguous()
        instance = instance.permute(0, 2, 3, 1).contiguous()
        gt_offset = gt_offset.permute(0, 2, 3, 1).contiguous()
        total_loss = 0.
        total_loss += self.confidence_loss(confidence, instance)*self.coeff[0]
        total_loss += self.offset_loss(offset, gt_offset, instance)*self.coeff[1]
        return total_loss