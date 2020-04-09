# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 13:39:38 2020

@author: 15652
"""

import numpy as np
import torch as t
from torch.utils.data import SubsetRandomSampler
from torch import nn
import json


# k折交叉验证
def k_fold(indices, k):
    stt = 0
    length = int(len(indices)/k)
    for _ in range(k):
        train_indices = np.concatenate((indices[:stt], indices[stt+length:]), axis=0)
        val_indices = indices[stt: stt+length]
        stt += length
        yield SubsetRandomSampler(train_indices), SubsetRandomSampler(val_indices)
        
        
# 计算交并比
def mIoU(A, B):
    assert len(A.size()) == 4 and len(B.size()) == 4, "the input of mIoU should be 4d"
    intersects = (A * B).sum(dim=(1, 2 ,3), dtype=t.float) # *相当于与运算
    unions = (A + B).sum(dim=(1, 2, 3), dtype=t.float) # +相当于或运算
    iou = t.zeros(intersects.size(), dtype=t.float)
    for i, union in enumerate(unions):
        if union == 0:
            iou[i] = 0
        else:
            iou[i] = intersects[i]/union
    miou = iou.mean().item()
    return miou
        

# 损失函数
class COLoss(nn.Module):
    def __init__(self, coeff=[1, 1, 1], bg_weight=0.4):
        super(COLoss, self).__init__()
        self.coeff = coeff
        self.bg_weight = bg_weight
            
    def compute_entropy(self, psuedo_distrib, distrib):
        psuedo_distrib = psuedo_distrib.gather(3, distrib.long())
        entropy = -t.log(psuedo_distrib)
        entropy[distrib==0] *= self.bg_weight  # 背景的交叉熵权重要小一点
        entropy = entropy.mean()
        return entropy
    
    def confidence_loss(self, confidence, binary):
        loss = self.compute_entropy(confidence, binary)
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
        confidence, offset = predictions
        instance, gt_offset = targets
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        offset = offset.permute(0, 2, 3, 1).contiguous()
        instance = instance.permute(0, 2, 3, 1).contiguous()
        gt_offset = gt_offset.permute(0, 2, 3, 1).contiguous()
        total_loss = 0.
        total_loss += self.confidence_loss(confidence, instance)*self.coeff[0]
        total_loss += self.offset_loss(offset, gt_offset, instance)*self.coeff[1]
        return total_loss


# 获得可用于TuSimple Benchmark的数据
def get_raw_points(predictions, opt):
    confidence, offset = predictions
    offset = offset.permute(0, 2, 3, 1).contiguous()
    coordinate = offset + opt._grid_location.to(opt.device)
    coordinate[:, :, :, 0].mul_(opt.resize_ratio[1])
    coordinate[:, :, :, 1].mul_(opt.resize_ratio[0])
    xy_values = []
    num_maps = confidence.size(1)
    for i in range(offset.size(0)):
        temp_x = []
        temp_y = []
        for j in range(1, num_maps):
            ins_mask = confidence[i, j].ge(opt.thresh_f)
            ins_mask = confidence[i, j].ge(opt.thresh_f)
            if ins_mask.sum().item() <= 1:
                continue
            temp_x.append(coordinate[i, :, :, 0][ins_mask].tolist())
            temp_y.append(coordinate[i, :, :, 1][ins_mask].tolist())
        xy_values.append([temp_x, temp_y])
    return xy_values


def interpolation(xy_values, h_samples, roots=None):
    results_data = []
    for i, (lanes, h) in enumerate(zip(xy_values, h_samples)):
        result = {'lanes': [], 'raw_file': '', 'run_time': 0.}
        for lane in np.rollaxis(np.array(lanes), 0, 2):
            lane_x = []
            num_points = len(lane[0])
            indices = np.searchsorted(lane[1], h)
            for index, y_sample in zip(indices, h):
                if index == 0 or index == num_points:
                    lane_x.append(-2)
                else:
                    x1, x2 = lane[0][index-1], lane[0][index]
                    y1, y2 = lane[1][index-1], lane[1][index]
                    x_value = ((y2 - y_sample)*x1 + (y_sample - y1)*x2)/(y2 - y1)
                    x_value = round(x_value, 2)
                    lane_x.append(x_value)
            result['lanes'].append(lane_x)
        if roots is not None:
            result['raw_file'] = roots[i]
        results_data.append(result)
    return results_data


def write_results(predictions, file, opt, roots=None):
    xy_values = get_raw_points(predictions, opt)
    results_data = interpolation(xy_values, [opt._h_samples]*len(xy_values), roots)
    for data in results_data:
        json.dump(data, file, separators=(', ', ': '))
        file.write('\n')