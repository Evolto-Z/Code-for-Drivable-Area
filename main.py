# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:58:33 2020

@author: 15652
"""


# 自建库
from config import Config
from utils import *
import models
from data import TuSimpleDataset

import torch as t
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm, trange
import time


opt = Config()


def new_collate_fn(batch):
    imgs_gts = []
    h_samples = []
    gt_lanes = []
    for item in batch:
        imgs_gts.append(item[: 2])
        h_samples.append(item[2])
        gt_lanes.append(item[3])
    return default_collate(imgs_gts), h_samples, gt_lanes


@t.no_grad()
def test(**kwargs):
    opt.parse(kwargs)
    
    # 配置模型
    model = getattr(models, opt.model)().eval()
    assert opt.pretrained_net_root is not None, "You should have a pretrained model first"
    model.load(opt.pretrained_net_root)
    model.to(opt.device)
    model.eval()
    
    # 准备好用来记录数据的json文件
    file = open(opt.json_pred, 'w')
    
    # 准备数据
    testset = TuSimpleDataset(opt, 'test')
    testloader = t.utils.data.DataLoader(testset,
                                         batch_size=opt.batch_size,
                                         shuffle=False,
                                         num_workers=opt.num_workers,
                                         drop_last=False)
    
    # 测试网络
    for inputs, roots in tqdm(testloader, ncols=80, desc='testing: '):
        stt = time.time()
        inputs = inputs.to(opt.device)
        _, predictions = model(inputs)
        write_results(predictions, stt, file, opt, roots)
    
    file.close()
    
    # 用TuSimple Benchmark评估结果
    print(LaneEval.bench_one_submit(opt.json_pred, opt.json_gt))


@t.no_grad()
def val(model, val_loader, vis):
    model.eval()
    
    accuracy_list = []
    # 验证训练效果
    for i, ((inputs, _,), h_samples, gt_lanes) in enumerate(tqdm(val_loader, ncols=80, desc='validating: ')):
        inputs = inputs.to(opt.device)
        _, predictions = model(inputs)
        xy_values = get_raw_points(predictions, opt)
        results_data = interpolation(xy_values, h_samples)
        pred_lanes = [result['lanes'] for result in results_data]
        accuracy = 0.
        for n, (pred, gt, h) in enumerate(zip(pred_lanes, gt_lanes, h_samples)):
            accuracy += (LaneEval.bench(pred, gt, h, 0)[0] - accuracy)/(n + 1)
        accuracy_list.append(accuracy)
            
    vis.plot('accuracy', sum(accuracy_list)/len(accuracy_list))


def train(**kwargs):
    opt.parse(kwargs)
    
    vis = Visualizer(env=opt.env, port=opt.vis_port)
    
    #配置模型
    model = getattr(models, opt.model)()
    if opt.pretrained_net_root:
        model.load(opt.pretrained_net_root)
    model.to(opt.device)
    
    # 配置损失函数和优化器
    criterion = LossFunc(opt).to(opt.device)
    optimizer = t.optim.Adam(model.parameters(), **opt.optim_config)
    
    # 准备数据
    train_val = TuSimpleDataset(opt, 'train')
    dataset_size = len(train_val)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    
    # 采用K折交叉验证
    fold = k_fold(indices, opt.k)
    
    for epoch in trange(opt.max_epoch, ncols=80, desc='training: '):
        try:
            (train_sampler, val_sampler) = next(fold)
        except StopIteration:
            np.random.shuffle(indices)
            fold = k_fold(indices, opt.k)
            (train_sampler, val_sampler) = next(fold)
        train_loader = DataLoader(train_val, 
                                  batch_size=opt.batch_size, 
                                  sampler=train_sampler, 
                                  num_workers=opt.num_workers,
                                  drop_last=True)
        val_loader = DataLoader(train_val,
                                batch_size=opt.batch_size, 
                                sampler=val_sampler,
                                collate_fn=new_collate_fn,
                                num_workers=opt.num_workers,
                                drop_last=True)
        # 训练部分
        model.train()
        for i, (inputs, targets, _, _) in enumerate(tqdm(train_loader, ncols=80, desc=('epoch %d: ' % (epoch+1)))):
            inputs = inputs.to(opt.device)
            targets = tuple([target_.to(opt.device) for target_ in targets])
            optimizer.zero_grad()
            predictions = model(inputs)
            loss0 = criterion(predictions[0], targets)
            loss1 = criterion(predictions[1], targets)
            loss = opt.gamma[0]*loss0 + opt.gamma[1]*loss1
            loss.backward()
            optimizer.step()
            if (i+1)%opt.update_freq == 0:
                vis.plot('loss', loss.item())
        if (epoch+1)%opt.save_freq == 0:
            model.save()

        # 验证部分
        val(model, val_loader, vis)


@t.no_grad()
def inference(**kwargs):
    opt.parse(kwargs)
    
    # 配置模型
    model = getattr(models, opt.model)().eval()
    assert opt.pretrained_net_root is not None, "You should have a pretrained model first"
    model.load(opt.pretrained_net_root)
    model.to(opt.device)
    model.eval()
    
    
    # 模式0: 推理图片
    if opt.infer_mode == 0:
        assert opt.infer is not None, "You should give an image first"
        img = Image.open(opt.infer)
        img_tensor = opt._trans(img).unsqueeze(0).to(opt.device)
        hook_handle = model.confidence2[0].register_forward_hook(forward_hook)
        _, predictions = model(img_tensor)
        xy_values = get_raw_points(predictions, opt)
        pred = interpolation(xy_values, [opt._h_samples])
        pred_lanes = pred[0]['lanes']
        pred_lanes_vis = [[(x, y) for (x, y) in zip(lane, opt._h_samples) if x >= 0] for lane in pred_lanes]
        img_vis = cv2.imread(opt.infer)
        for lane in pred_lanes_vis:
            cv2.polylines(img_vis, np.int32([lane]), isClosed=False, color=(0,0,255), thickness=2)
        cv2.imshow('inference', img_vis)
        cv2.waitKey(0)
        cv2.destroyWindow('inferece')
        hook_handle.remove()
        
    # 模式1: 推理视频
    if opt.infer_mode == 1:
        assert opt.infer is not None, "You should give a video first"
        cap = cv2.VideoCapture(opt.infer)
        fps = 20 # 与输入视频相同
        img_size = (1280,720) # 与输入视频相同
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        saver = cv2.VideoWriter('temp/video/inference.avi', fourcc, fps, img_size)
        while True:
            ret, frame = cap.read()
            preTime = time.time()
            if ret == False:
                break
            img = Image.fromarray(frame)
            img_tensor = opt._trans(img).unsqueeze(0).to(opt.device)
            _, predictions = model(img_tensor)
            xy_values = get_raw_points(predictions, opt)
            pred = interpolation(xy_values, [opt._h_samples])
            pred_lanes = pred[0]['lanes']
            pred_lanes_vis = [[(x, y) for (x, y) in zip(lane, opt._h_samples) if x >= 0] for lane in pred_lanes]
            curTime = time.time()
            sec = curTime - preTime
            fps = round(1/(sec), 1)
            s = "FPS : "+ str(fps)
            cv2.putText(frame, s, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
            for lane in pred_lanes_vis:
                cv2.polylines(frame, np.int32([lane]), isClosed=False, color=(0,0,255), thickness=2)
            cv2.imshow('inferece', frame)
            saver.write(frame)
            if cv2.waitKey(1) == 27:  # 键盘esc键所对应的ascii码
                break
        cap.release()
        cv2.destroyWindow('inferece')


if __name__ == '__main__':
    import fire
    fire.Fire()