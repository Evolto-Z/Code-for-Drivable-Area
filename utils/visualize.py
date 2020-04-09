# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:19:26 2020

@author: 15652
"""


import visdom
import numpy as np


class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)
        self.index = {}
        
    def reinit(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)
        self.index = {}
        return self
        
    def plot_many(self, d):
        for k, v in d.items():
            self.plot(k, v)
            
    def plot(self, name, y):
        x = self.index.get(name, 0)
        self.vis.line(
                X=np.array([x]),
                Y=np.array([y]),
                win=name,
                opts={'title': name},
                update=None if x==0 else 'append'
            )
        self.index[name] = x + 1
        
    def show_many(self, d):
        for k, v in d.items():
            self.show(k, v)
            
    def show(self, name, img):
        if len(img.size()) == 3:
            img.unsqueeze(0)
        self.vis.images(
                img.cpu(),
                win=name,
                opts={'title': name}
            )
        
    def __getattr__(self, name):
        return getattr(self.vis, name)