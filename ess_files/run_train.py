#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import os
import sys, os
import torch
from sapsal.cINN_config import read_config_from_file
from sapsal.data_loader import DataLoader
from sapsal.execute import train_network #test
from sapsal.tools.logger import Logger

from argparse import ArgumentParser
from time import time

# # from cINN.test_execute import train_network

GPU_MAX_LOAD = 0.6           # Maximum compute load of GPU allowed in automated selection
GPU_MAX_MEMORY = 0.6         # Maximum memory load of GPU allowed in automated selection
GPU_WAIT_S = 600             # Time (s) to wait between tries to find a free GPU if none was found
GPU_ATTEMPTS = 10            # Number of times to retry finding a GPU if none was found
GPU_EXCLUDE_IDS = [] # List of GPU IDs that are to be ignored when trying to find a free GPU, leave as empty list if none
VERBOSE = True
MAX_EPOCH = 500 #1000

LOG_MODE = True     # Print both console and log # this does not set to save log!

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('config_file', help="Run with specified config file as basis.")
    parser.add_argument('-d','--device', required=False, default=None, help="Device")
    parser.add_argument('-L','--log', required=False, default=True, help="Save logfile T/F")
    parser.add_argument('--max_epoch', required=False, default=None, help="Max epoch for training (default=500)")
    parser.add_argument('-r','--resume', required=False, default=False, action='store_true', help="Resume training using checkpoint (T/F)")
    args = parser.parse_args()
    
    print("Start main (run_train)")
    
    if args.log =='True' or  args.log=='1':
        savelog=True
    elif args.log is not True:
        savelog = False
    else:
        savelog = True
    
    if args.max_epoch is not None:
        MAX_EPOCH = int(args.max_epoch)
        
    resume = args.resume
    
        
    # if savelog:
    #     logfile = os.path.basename(args.config_file).replace('.py','_train.log').replace('c_','')
    #     sys.stdout = Logger(logfile, log_mode = LOG_MODE)
        
        
    # config_file = sys.argv[1]
    config_file = args.config_file
    c = read_config_from_file(config_file)
    print(config_file) 
    
    if resume:
        print("Resume training if checkpoint exists")
    
    # creat output directory and move log file
    if savelog:
        # sys.stdout.close()  
        # sys.stdout = sys.__stdout__  # stdout 복원
        
        # make savedir
        logpath = os.path.dirname(c.filename)+'/'
        if not os.path.exists(logpath):
            os.system('mkdir -p '+logpath)
            
        # new_logfile = logpath + logfile
        # os.system(f'mv {logfile} {new_logfile}')  # move log file
        # logfile = new_logfile
        
        logfile = logpath.replace('/','.') + os.path.basename(c.filename) +'_train.log'
        
        # sys.stdout = Logger(logfile, log_mode=LOG_MODE, mode="a") # continue logging
        logger = Logger(logfile, log_mode=LOG_MODE, mode="a")
        
    if os.environ.get("CUDA_LAUNCH_BLOCKING")==1: # This is only for debugging. do not use   
        print("CUDA_LAUNCH_BLOCKING =", os.environ.get("CUDA_LAUNCH_BLOCKING"))
        
    data_loading_start=time()
    astro = DataLoader(c, update_rescale_parameters=True)
    print(f"[Time for data loading: {time() - data_loading_start:.2f}s]")
    
    # if len(sys.argv)==3:
    if args.device is not None:
        # device = sys.argv[2]
        device = args.device
        print('Change device from %s to %s'%(c.device, device))
        c.device = device

         
    if 'cuda' in c.device:
         import GPUtil
         DEVICE_ID_LIST = GPUtil.getFirstAvailable(maxLoad=GPU_MAX_LOAD,
                                                   maxMemory=GPU_MAX_MEMORY,
                                                   attempts=GPU_ATTEMPTS,
                                                   interval=GPU_WAIT_S,
                                                   excludeID=GPU_EXCLUDE_IDS,
                                                   verbose=VERBOSE)
         DEVICE_ID = DEVICE_ID_LIST[0]
         c.device = 'cuda:{:d}'.format(DEVICE_ID)
    
    elif 'mps' in c.device: # request to use MAC GPU
        # CPU will be used if MPS is not available, assuming you are running on MAC
        if torch.backends.mps.is_available() * torch.backends.mps.is_built():
            c.device = 'mps'
        else:
            c.device = 'cpu'
    else:
        c.device = 'cpu'
                    
         
    
    print("Device: ",c.device)
    # _ = torch.Tensor([0]).to(c.device)
    astro.device = c.device
    
    print("Maximum training epoch: ", MAX_EPOCH)
    
    # Run train_network function in cINN.execute
    if c.prenoise_training == True:
        # from sapsal.execute import train_prenoise_network
        # train_prenoise_network(c, data=astro, max_epoch=MAX_EPOCH)
        train_network(c, data=astro, max_epoch=MAX_EPOCH, resume=resume) # now train_prenoise_network is combined with train_network
    elif c.use_flag == True:
        from sapsal.execute import train_flag_network
        train_flag_network(c, data=astro, max_epoch=MAX_EPOCH)
    elif c.wavelength_coupling == True:
        from sapsal.execute import train_wc_network
        train_wc_network(c, data=astro, max_epoch=MAX_EPOCH)
    else:
        if c.domain_adaptation:
            if c.da_without_discriminator:
                from sapsal.execute import train_network_DAwoD
                train_network_DAwoD(c, data=astro, max_epoch=MAX_EPOCH)
        else:
            train_network(c, data=astro, max_epoch=MAX_EPOCH)
    
    print("Finished training: %s"%config_file)
        
    # move log file to path
    if savelog:
        # sys.stdout.close()  # close log file
        # sys.stdout = sys.__stdout__ # stop logging
        logger.close()
        
        # logpath = os.path.dirname(c.filename)+'/'
        new_logfile = logpath + os.path.basename(c.filename) +'_train.log'
        os.system("mv {} {}".format(logfile, new_logfile)  )
