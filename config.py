# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 11:14:57 2020

@author: 15652
"""


import warnings
import torch as t
import torchvision.transforms as T


class Config(object):
    # 配置visdom
    env = 'Project_Delta'
    vis_port = 8097
    update_freq = 32 # N个batch更新一次
    
    # 路径和文件名配置
    dataset_root = 'd:/Finale/Project/data/TuSimple/'
    pretrained_net_root = None
    json_pred = 'test_pred.json'
    json_gt = 'test_gt.json'
    
    # 系统参数配置
    use_gpu = True
    num_workers = 0
    
    # 超参数配置
    batch_size = 1
    optim_config = dict(
            lr = 0.001,
            betas = (0.9, 0.999),
            weight_decay = 0.0001
        )
    
    # 损失函数参数配置
    bg_weight = 0.4  # 背景的权重
    coeff = [2, 1]
    gamma = [1, 3]
    
    # 其他参数配置
    pre_img_size = (720, 1280)
    resize_size = (576, 1024)
    img_size = (72, 128)
    resize_ratio = (pre_img_size[0]//img_size[0], pre_img_size[1]//img_size[1])
    max_epoch = 128
    k = 8 # k折交叉验证
    thresh = 0.25 # 预测概率大于该值即为对应实例
    save_freq = 32 # N个epoch保存一次
    _grid_location = t.zeros((1, img_size[0], img_size[1], 2), dtype=t.float)
    for _i in range(img_size[0]):
        for _j in range(img_size[1]):
            _grid_location[0, _i, _j, 0] = _j
            _grid_location[0, _i, _j, 1] = _i
    _h_samples = list(range(160, 720, 10))
    
    _trans = T.Compose([
        T.Resize(resize_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # 模型名
    model = 'NetDelta'
    
    # inference模式
    infer_mode = 0
    infer = None
    
    # 在命令行中可以对默认配置进行修改
    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn('Not have attribute %s' % k)
                continue
            setattr(self, k, v)
        
        if self.use_gpu == True and t.cuda.is_available():
            self.device = t.device('cuda')
        elif self.gpu == True:
            print('Sorry, you can only use cpu.')
            self.device == t.device('cpu')
        else:
            self.device == t.device('cpu')
        
        print('User Config: ')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))