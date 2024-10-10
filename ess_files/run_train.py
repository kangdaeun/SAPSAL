#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
from cINN_set.cINN_config import read_config_from_file
from cINN_set.data_loader import DataLoader
from cINN_set.execute import train_network #test
from cINN_set.execute import train_prenoise_network
from cINN_set.execute import train_flag_network
from cINNset.execute import train_wc_network
# # from cINN.test_execute import train_network

GPU_MAX_LOAD = 0.1          # Maximum compute load of GPU allowed in automated selection
GPU_MAX_MEMORY = 0.1         # Maximum memory load of GPU allowed in automated selection
GPU_WAIT_S = 600             # Time (s) to wait between tries to find a free GPU if none was found
GPU_ATTEMPTS = 10            # Number of times to retry finding a GPU if none was found
GPU_EXCLUDE_IDS = [] # List of GPU IDs that are to be ignored when trying to find a free GPU, leave as empty list if none
VERBOSE = True
MAX_EPOCH = 200 #1000

if __name__ == '__main__':
    #config_file = 'config_rncpr01_x7_y12_train_example'
    config_file = sys.argv[1]
    c = read_config_from_file(config_file)
    print(config_file)
    
    astro = DataLoader(c, update_rescale_parameters=True)

    if len(sys.argv)==3:
         device = sys.argv[2]
         print('Change device from %s to %s'%(c.device, device))
         c.device = device
    elif 'cuda' in c.device:
         import GPUtil
         DEVICE_ID_LIST = GPUtil.getFirstAvailable(maxLoad=GPU_MAX_LOAD,
                                                   maxMemory=GPU_MAX_MEMORY,
                                                   attempts=GPU_ATTEMPTS,
                                                   interval=GPU_WAIT_S,
                                                   excludeID=GPU_EXCLUDE_IDS,
                                                   verbose=VERBOSE)
         DEVICE_ID = DEVICE_ID_LIST[0]
         c.device = 'cuda:{:d}'.format(DEVICE_ID)
    
    if 'mps' in c.device: # request to use MAC GPU
        # CPU will be used if MPS is not available, assuming you are running on MAC
        if torch.backends.mps.is_available() * torch.backends.mps.is_built():
            c.device = 'mps'
        else:
            c.device = 'cpu'
                    
         
    
    print("Device: ",c.device)
    _ = torch.Tensor([0]).to(c.device)
    astro.device = c.device

    # Run train_network function in cINN.execute
    if c.prenoise_training == True:
        train_prenoise_network(c, data=astro, max_epoch=MAX_EPOCH)
    elif c.use_flag == True:
        train_flag_network(c, data=astro, max_epoch=MAX_EPOCH)
    elif c.wavelength_coupling == True:
        train_wc_network(c, data=astro, max_epoch=MAX_EPOCH)
    else:
        train_network(c, data=astro, max_epoch=MAX_EPOCH)
