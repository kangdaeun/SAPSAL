#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 12:02:21 2023

@author: daeun


Tools for training Simulation Gap network : FTransformNet

"""

# Convergence 
# CONV_CUT = 1e-3
CONV_CUT = 8e-4
N_CONV_CHECK = 20

# Divergence
DIVG_CHUNK_SIZE = 7
N_DIVG_CHECK = 35
DIVG_CRI = 0



# import torch

# def get_loaders(feature_test, feature_train, batch_size):

    
#     test_loader = torch.utils.data.DataLoader( 
#                 torch.utils.data.TensorDataset(torch.Tensor(feature_test)),
#                 batch_size=batch_size, shuffle=True, drop_last=True)
             
#     train_loader = torch.utils.data.DataLoader(
#         torch.utils.data.TensorDataset(torch.Tensor(feature_train)),
#         batch_size=batch_size, shuffle=True, drop_last=True)
        
#     return test_loader, train_loader


# import torch.nn
# import torch.optim
# from torch.nn.functional import avg_pool2d, interpolate
# from torch.autograd import Variable
# from time import time
# import os
# # import numpy as np
# import tqdm
# from ..viz import *
# from . import train_tools


# from scipy import stats




 