# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 20:12:54 2020

@author: 15652
"""

import cv2
import os
from tqdm import trange
from tqdm import tqdm
from glob import glob


img_root = '../temp/img/'  # 图片存储路径
video_root = '../temp/video/1.avi' # 视频存储路径及视频名
file_ls = glob(os.path.join(img_root, '*'))
img_ls = []
for file in file_ls:
    temp = glob(os.path.join(file+'/', '*.jpg'))
    temp.sort(key=lambda img: int(img.split('\\')[-1].split('.')[-2]))
    img_ls.extend(temp)
fps = 20 # 帧率一般选择20-30
num = 101 # 图片数 + 1
img_size = (1280,720)

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
videoWriter = cv2.VideoWriter(video_root, fourcc, fps, img_size)

for img in tqdm(img_ls, ncols=80):
    frame = cv2.imread(img)
    videoWriter.write(frame)

videoWriter.release()