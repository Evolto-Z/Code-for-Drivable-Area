# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 20:12:54 2020

@author: 15652
"""


import cv2
import os
from tqdm import trange


img_root = '../temp/img/'  # 图片存储路径
video_root = '../temp/video/1.avi' # 视频存储路径及视频名
fps = 20 # 帧率一般选择20-30
num = 21 # 图片数 + 1
img_size = (1280,720)

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
videoWriter = cv2.VideoWriter(video_root, fourcc, fps, img_size)

for i in trange(1, num, ncols=80):
    im_name = os.path.join(img_root, str(i)+'.jpg')
    frame = cv2.imread(im_name)
    videoWriter.write(frame)

videoWriter.release()