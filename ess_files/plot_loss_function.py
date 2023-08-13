#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 19:11:24 2020

@author: daeun

plot loss function after training
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
from cINN_set.cINN_config import *
from cINN_set.viz import plot_loss_curve
from cINN_set.viz import plot_loss_curve_2types

#config_file = 'config_rncpr01_x7_y12_02'
config_file = sys.argv[1]    
c = read_config_from_file(config_file)
print(config_file)

#%%
figname = (c.filename+'_Loss_plot.pdf')#.replace('output/','output/tmp_storage/')
loss_file = (c.filename+'_loss_history.txt')#.replace('output/','output/tmp_storage/')


# epoch, loss, test_loss = np.loadtxt(loss_file, unpack=True, usecols=(0,1,3))
epoch_loss_history = np.genfromtxt('output/test_train.pt_loss_history.txt', names=True)
#%%

# fig, ax = plot_loss_curve(epoch, loss, test_loss, c = c, figname=figname, title=os.path.basename(c.config_file).replace('.py',''),
#                           xrange=None, yrange = c.loss_plot_yrange)


fig, ax = plot_loss_curve_2types(epoch_loss_history['Epoch'],
                                 epoch_loss_history['Loss_train_mn'],
                                 epoch_loss_history['Loss_test_mn'],
                                 epoch_loss_history['Loss_train_mdn'],
                                 epoch_loss_history['Loss_test_mdn'],
                                 c = c, figname=figname, title=os.path.basename(c.config_file).replace('.py',''),
                          xrange=None, yrange = c.loss_plot_yrange)