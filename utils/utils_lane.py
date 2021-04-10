# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 13:39:38 2020

@author: 15652
"""

import numpy as np
import torch as t
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms as T
from matplotlib import pyplot as plt
import json
import time


def forward_hook(model, in_one, out_one):
    img = in_one[0].squeeze(0)
    _, img = img.max(dim=0)
    img = img.type(t.uint8)
    img = T.ToPILImage()(img.cpu())
    plt.imshow(img)
    plt.show()


# k折交叉验证
def k_fold(indices, k):
    stt = 0
    length = int(len(indices)/k)
    for _ in range(k):
        train_indices = np.concatenate((indices[:stt], indices[stt+length:]), axis=0)
        val_indices = indices[stt: stt+length]
        stt += length
        yield SubsetRandomSampler(train_indices), SubsetRandomSampler(val_indices)


# 获得可用于TuSimple Benchmark的数据
def get_raw_points(predictions, opt):
    confidence, offset = predictions['confidence'], predictions['offset']
    offset = offset.permute(0, 2, 3, 1).contiguous()
    coordinates = offset + opt._grid_location.to(opt.device)
    coordinates[:, :, :, 0].mul_(opt.resize_ratio[1])
    coordinates[:, :, :, 1].mul_(opt.resize_ratio[0])
    xy_values = []
    num_maps = confidence.size(1)
    for i in range(offset.size(0)):
        xy_value = []
        for j in range(1, num_maps):
            prob = confidence[i, j]
            coordinate = coordinates[i]
            vals, indices = t.max(prob, dim=1, keepdim=True)
            mask = t.zeros_like(prob).to(opt.device)
            mask.scatter_(1, indices, vals.ge(opt.thresh).float())
            if mask.sum().item() < 6:
                continue
            xy_value.append(coordinate[mask.bool()].tolist())
        xy_values.append(xy_value)
    return xy_values


def interpolation(xy_values, lane_ys, roots=None):
    results = []
    if not isinstance(lane_ys[0], list):
        lane_ys = [lane_ys]*len(xy_values)
    for i ,(xy_value, lane_y) in enumerate(zip(xy_values, lane_ys)):
        result = {'lanes': [], 'raw_file': '', 'run_time': 0.}
        for lane in xy_value:
            lane_x = [-2]*len(lane_y)
            lane_y_pred = list(map(lambda point: point[1], lane))
            num_points = len(lane_y_pred)
            indices = np.searchsorted(lane_y_pred, lane_y)
            count = 0
            for n, (index, y_sample) in enumerate(zip(indices, lane_y)):
                if index == 0:
                    continue
                elif index == num_points:
                    break
                else:
                    x1, y1 = lane[index-1]
                    x2, y2 = lane[index]
                    x = ((y2 - y_sample)*x1 + (y_sample - y1)*x2)/(y2 - y1)
                    x = round(x)
                    lane_x[n] = x
                    count += 1
            if count >= 6:
                result['lanes'].append(lane_x)
        if roots is not None:
            result['raw_file'] = roots[i]
        results.append(result)
    return results


def write_results(predictions, stt, file, opt, roots=None):
    xy_values = get_raw_points(predictions, opt)
    results = interpolation(xy_values, opt._h_samples, roots)
    run_time = (time.time() - stt)/len(results)
    for data in results:
        data['run_time'] = round(run_time*1000, 2)
        json.dump(data, file, separators=(', ', ': '))
        file.write('\n')