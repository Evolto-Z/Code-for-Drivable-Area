# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:21:57 2020

@author: 15652
"""


import torch as t
from torch.utils import data
from glob import glob
import os
import json
from PIL import Image


class TuSimpleDataset(data.Dataset):
    def __init__(self, opt, flag='train'):
        assert flag == 'train' or flag == 'test', "flag in TuSimpleDataset should be 'train', 'test'"
        self.dataset_root = opt.dataset_root
        self.flag = flag
        self.img_size = opt.img_size
        self.resize_ratio = opt.resize_ratio
        self.json_ls = glob(os.path.join(self.dataset_root, self.flag+'/', '*.json'))
        self.raw_data = []
        for json_ in self.json_ls:
            self.raw_data.extend([json.loads(line) for line in open(json_)])
        self.trans = opt._trans

    def make_ground_truth(self, x_label, y_label):        
        gt_offset = t.zeros((2, self.img_size[0], self.img_size[1]), dtype=t.float)
        gt_instance = t.zeros((1, self.img_size[0], self.img_size[1]), dtype=t.uint8)
        lane_id = 0
        for lane_x in x_label:
            pre_x_index, pre_y_index = None, None
            lane_id += 1
            for i, x_ in enumerate(lane_x):
                if x_ >= 0:
                    x_index, y_index = int(x_), int(y_label[i])
                    gt_offset[0, y_index, x_index] = x_ - x_index
                    gt_offset[1, y_index, x_index] = y_label[i] - y_index
                    gt_instance[0, y_index, x_index] = lane_id
                    while pre_x_index is not None and pre_y_index is not None:
                        delta_x, delta_y = 0, 0
                        if pre_x_index < x_index:
                            delta_x = 1
                        elif pre_x_index > x_index:
                            delta_x = -1
                        if pre_y_index < y_index:
                            delta_y = 1
                        elif pre_y_index > y_index:
                            delta_y = -1
                        if delta_x == 0 and delta_y == 0:
                            break
                        pre_x_index += delta_x
                        pre_y_index += delta_y
                        gt_instance[0, pre_y_index, pre_x_index] = lane_id
                    pre_x_index, pre_y_index = x_index, y_index
        return gt_instance, gt_offset
    
    def __getitem__(self, index):
        piece = self.raw_data[index]
        img_root = piece['raw_file']
        pil_img = Image.open(os.path.join(self.dataset_root, self.flag+'/', img_root))
        x_label = t.Tensor(piece['lanes'])/self.resize_ratio[1]
        y_label = t.Tensor(piece['h_samples'])/self.resize_ratio[0]
        img = self.trans(pil_img)
        gt = self.make_ground_truth(x_label, y_label)
        if self.flag == 'train':
            return img, gt, piece['h_samples'], piece['lanes']
        else:
            return img, img_root
    
    def __len__(self):
        return len(self.raw_data)