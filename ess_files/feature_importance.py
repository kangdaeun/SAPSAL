#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:59:52 2022

@author: daeun

Feature importance test

User setup
- config / network
- feature important test section

Todo :
    - FI_test/ 폴더가 꼭 필요한 것인지. 안만들면 얼마나 복잡해지는지. 혹은 kwarg로 받아야하는지
    - 파일 이름: 타겟 특성별로 구분하거나 사이즈 바꾸거나 사이즈 방식을 바꾸거나 등등을 고려할 수 있는 네이밍
    - 함수화
    - 조건에 따라 테스트 할 네이밍 바꿔주는 함수를 만들어야 할듯. (지금 메인에 대문자로 적용되는 것들.)

"""

import numpy as np
# import matplotlib.pyplot as plt

import os, sys
# import matplotlib.cm as cm
# from pathlib import Path
from time import time
from argparse import ArgumentParser

# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# import matplotlib.patheffects as PathEffects

# import matplotlib.colors as clr
# import matplotlib.ticker as ticker

import torch
from astropy.table import Table, vstack, Column
from astropy.io import ascii

from sapsal.cINN_config import read_config_from_file
from sapsal.data_loader import DataLoader

import copy

# from tqdm import tqdm


FI_SIMPLE_GROUP_SIZE = 10        # Size of feature (simple linear bin)
FI_SIMPLE_GROUP_OVERLAP = 5     # Size of overalpping bin
FI_INTERVAL_FILEPATH = None     # Path to desginate interval files. 

# OUTPUT_DIR = ""         # Path to output directory
# OUTPUT_SUFFIX = "FI_"        # Optional suffix for all output files (can change by option argument)


logTeff_condition = {
    0: 'VARI <= np.log10(3850)',  # S, NG, D (D~all)
    1: 'VARI <= np.log10(4590)', # S, NG (D=all)
    2: 'np.logical_and( VARI >= np.log10(3900), VARI <= np.log10(5110) )', # S, NG
    3: 'VARI >= np.log10(5150)', # S, NG
    4: 'VARI >= np.log10(4600)', # S, NG
    5: 'np.logical_and( VARI >= np.log10(5150), VARI <= np.log10(6000) )', # S, NG (G type)
}
    
DATA_CONDITION_DIC = {}     # Set the condition of models to use: empty dic mean using all test data
                            # key must have one of the c.x_names
DATA_CONDITION_DIC['logTeff'] = logTeff_condition[5]

COND_SUFFIX = '_logTeff_5'    # suffix for condition. if you dont use condition keep ''

DEVICE = 'cuda'              # Device to run search on, either 'cuda' or 'cpu'
GPU_MAX_LOAD = 0.1           # Maximum compute load of GPU allowed in automated selection
GPU_MAX_MEMORY = 0.1         # Maximum memory load of GPU allowed in automated selection
GPU_WAIT_S = 600             # Time (s) to wait between tries to find a free GPU if none was found
GPU_ATTEMPTS = 10            # Number of times to retry finding a GPU if none was found
GPU_EXCLUDE_IDS = [] # List of GPU IDs that are to be ignored when trying to find a free GPU, leave as empty list if none

VERBOSE = True           # Switch for progress messages



# Posterior and MAP setting
N_PRED = 4096           # latent variable sampling for one model
MAP_KWARG = {'bw_method':'silverman', 
             'n_grid':1024, 
             'use_percentile': None,
             'plot':False }

POST_GROUP_SIZE = 250  # n(obs) per one iteration

 
#%%

def caluclate_FMAP(interval, y_data, c, exp=None,
                 N_pred=4096, post_group=200, 
                 **kwarg
                 ):

    # i_start, i_end = interval
    n_data = len(y_data)  
    
    # Permute Y matrix for examine interval
    y_pert = y_data.copy()
    for k in interval:
        y_pert[:, k] = y_pert[:, k][np.random.permutation(N_data)]
        
    
    if exp is None:
        exp = c.import_expander()
        
    # Compute posterior distributions for all given data
    n_group = np.ceil(n_data/post_group).astype(int)
    for i_group in range(n_group):
        y_group = y_pert[i_group*post_group:(i_group+1)*post_group]
        
        # posterior per group
        post_list = exp.get_posterior(y_group, c, N=N_pred, use_group=True, group=post_group,
                                      return_llike=False, quiet=True)
        if i_group==0:
            fi_post = post_list
        else:
            fi_post = np.vstack([fi_post, post_list])
            
    # Compute kde MAP estimates
    fi_map_list = []
    for post in fi_post:
        fi_map_list.append( exp.calculate_map(post, c, **kwarg) )
    fi_map_list = np.array(fi_map_list)
    
    return fi_map_list
        
  
def make_even_interval(y_names, fi_group_size=FI_SIMPLE_GROUP_SIZE, fi_overlap=0):
    
    # Define intervals to examine : [ (start, end), (start, end), (start, end)....]
    
    interval_index_list = []
    interval_ynames_list = []
    
    # overlap should not exceed group size
    if fi_overlap >= fi_group_size:
        print("Overlap >= group size. Automatically adjust overlap=np.floor(group_size*0.5)")
        fi_overlap = np.floor( 0.5* fi_group_size).astype(int)
    
    n_f = len(y_names)
    fi_group_size = int(fi_group_size)
    # if y_names is None:
    #     y_names = np.arange(0, n_f)
    y_names = np.array(y_names)
    
    i_start = 0; i_end = fi_group_size
    
    while i_start < n_f:

        interval_index_list.append( list( np.arange(i_start, i_end)))
        interval_ynames_list.append(  list(y_names[list( np.arange(i_start, i_end))])   )

        i_start = i_end - fi_overlap
        i_end = i_start + fi_group_size
        if i_end >= n_f:
            i_end = n_f
            if (i_end-i_start) > fi_overlap:
                interval_index_list.append( list( np.arange(i_start, i_end)))
                interval_ynames_list.append(  list(y_names[list( np.arange(i_start, i_end))])   )
            break
            
    return interval_index_list, interval_ynames_list   
    
        


##########
## MAIN ##
##########

if __name__=='__main__':
    
    # Parse optional command line arguments
    parser = ArgumentParser()
    # parser.add_argument("-c", "--config", dest="config_file", default='./config.py',
    #                     help="Run with specified config file as basis.")
    
    parser.add_argument('config_file', help="Run with specified config file as basis.")
    parser.add_argument('-d','--device', required=False, default='cuda', help="device: cuda or cpu")
    # parser.add_argument('-s','--suffix', required=False, default=None, help="Output suffix")
    # parser.add_argument('-o','--outputdir', required=False, default=None, help="Output directory")
    
    # Import default config
    args = parser.parse_args()

    # Import default config
    config_file = args.config_file
    c = read_config_from_file(config_file)
    
    if 'cuda' in args.device:
        import GPUtil
        DEVICE_ID_LIST = GPUtil.getFirstAvailable(maxLoad=GPU_MAX_LOAD,
                                                  maxMemory=GPU_MAX_MEMORY,
                                                  attempts=GPU_ATTEMPTS,
                                                  interval=GPU_WAIT_S,
                                                  excludeID=GPU_EXCLUDE_IDS,
                                                  verbose=VERBOSE)
        DEVICE_ID = DEVICE_ID_LIST[0]
        c.device = 'cuda:{:d}'.format(DEVICE_ID)
    else:
        c.device = 'cpu'
    
    _ = torch.Tensor([0]).to(c.device)
    print("==================== CINN NETWORK SETTING =================")
    print("cINN_config:", config_file)
    print("# of parameters:", c.x_dim)
    print("# of observables:", c.y_dim_in)
    print("Database:", c.tablename)
    print("using device:", c.device)
    print("===========================================================")
    
    network_name = os.path.basename(c.filename).replace('.pt','')

    exp = c.import_expander()
    
    #%%
    """
    Set data to examine and make reference values
        default: using all test set (defined by test_frac)
        only models with specific conditions
    """
    
    astro = DataLoader(c)
    test_set, train_set = astro.get_splitted_set(rawval=True, smoothing = False, smoothing_sigma = None,
                                                 normalize_flux=c.normalize_flux, 
                                                 normalize_total_flux=c.normalize_total_flux, 
                                                 normalize_mean_flux=c.normalize_mean_flux
                                                 )
    param_test = test_set[0]; obs_test = test_set[1]
    param_table = Table(param_test, names=c.x_names)
    
    # READ MAP for the reference error
    map_table = ascii.read(c.filename+'_MAP.dat',  format='commented_header', delimiter='\t')
    map_list = np.array(map_table[c.x_names]).view(float).reshape(-1,len(c.x_names))
    
    
    # Check if you have special condition for test models to evaluate
    roi_cond = np.array([True]*len(param_test))
    for key, st in DATA_CONDITION_DIC.items():
        if key in c.x_names:
            condi = st.replace('VARI', 'param_table["%s"]'%key) # condition for the parameter
            roi_cond *= eval(condi)
            
    
    # filter conditions
    param_data = param_test[roi_cond]
    obs_data = obs_test[roi_cond]
    map_list = map_list[roi_cond]
    
    N_data = obs_data.shape[0]
    print("# of test models: %d"%N_data)
    
    # Reference RMSE 
    rmse_ind_ref = np.sqrt(np.mean((param_data - map_list)**2, axis = 0)) 
    rmse_tot_ref = np.sqrt(np.mean(np.sum((param_data - map_list)**2, axis=1))) 
    print("RMSE (reference)")
    print("RMSE for each param:", rmse_ind_ref,)
    print("RMSE total:",rmse_tot_ref)
    
    #%%
    
    # Define intervals to examine : [ [~,~,~], [~,~,~,~,]... ]
    # Or read intervals from file
    # if FI_INTERVAL_FILEPATH is not None:
    #     if os.path.exists(FI_INTERVAL_FILEPATH):
                  
    if  FI_SIMPLE_GROUP_SIZE is not None:
        fi_index_list, fi_name_list = make_even_interval(y_names=c.y_names, fi_group_size=FI_SIMPLE_GROUP_SIZE, fi_overlap=FI_SIMPLE_GROUP_OVERLAP)
    
    N_tests = len(fi_index_list)
    
    # FI_SIMPLE_GROUP_SIZE = 9 # top of the code
    # if FI_SIMPLE_GROUP_SIZE is not None:
    #     FI_FEATURE_LABELS = ["%i" % g for g in range(np.ceil(len(c.y_names)/FI_SIMPLE_GROUP_SIZE).astype(int))]          # List of str with the label to adhere to every test. If None, will be automatically infered from feature list and column names.
        
    #     FI_FEATURE_LIST_INDEX = [list(np.arange(n * FI_SIMPLE_GROUP_SIZE, (n + 1) * FI_SIMPLE_GROUP_SIZE, 1)) for n in range(len(FI_FEATURE_LABELS))]            # List of lists with feature column indices to test feature importance for. If None tests all of them individually.
    #     if len(c.y_names)%FI_SIMPLE_GROUP_SIZE > 0:
    #         FI_FEATURE_LIST_INDEX[-1] = FI_FEATURE_LIST_INDEX[-1][:len(c.y_names)%FI_SIMPLE_GROUP_SIZE]
    #     FI_FEATURE_LIST_NAMES = [c.y_names[n*FI_SIMPLE_GROUP_SIZE:(n+1)*FI_SIMPLE_GROUP_SIZE] for n in range(len(FI_FEATURE_LABELS))]
    # FI_PLOT_MAP = False 
    #%%
    
    """
    Set output file path 
    """
    savedir = os.path.dirname(c.filename)+'/'
    if not os.path.exists(savedir):
        print("make savedir: %s"%savedir)
        os.system("mkdir %s"%savedir)
    
    if  FI_SIMPLE_GROUP_SIZE is not None:
        suffix = '_Nb_{}_Nov_{}'.format(FI_SIMPLE_GROUP_SIZE, FI_SIMPLE_GROUP_OVERLAP)
    
    fi_filename = savedir + 'FI_' + network_name + suffix + COND_SUFFIX +  '.dat'
    print("Filename: %s"%(fi_filename.replace(savedir,"")))
    
    
    feature_importance = np.zeros(shape=(N_tests, len(c.x_names)+1))
    div_i_range = np.zeros(shape=(N_tests, 2))
    
    if os.path.exists(fi_filename):
        fi_table = ascii.read(fi_filename, format='commented_header', delimiter='\t')
        n_start = len(fi_table)
    else:
        n_start = 0
    
    col_names = ['i_start', 'i_end']+['Total'] + c.x_names
    header = ''
    for i in col_names:
        header=header+'{}\t'.format(i)
    
    
    #%%
    print("# of divisions: %d"%N_tests)
    print("# of divisions to run: %d"%(N_tests-n_start))
    
    t_start = time()
    t_mid = t_start
    
    for n in range(n_start, N_tests):
        print("%d th division"%n)
        
        fi_map_list = caluclate_FMAP( fi_index_list[n], obs_data, c, exp=exp,
                         N_pred=N_PRED, post_group=POST_GROUP_SIZE, **MAP_KWARG)
        
    
        # Compute RMSE
        rmse_ind_n = np.sqrt(np.mean((param_data - fi_map_list)**2, axis = 0))           # RMSE of each param
        rmse_tot_n = np.sqrt(np.mean(np.sum((param_data - fi_map_list)**2, axis=1)))     # RM (total SE: sum of 3 square errors) 
    
        
        tot_fi = rmse_tot_n / rmse_tot_ref  # Total importance
        param_fi = rmse_ind_n / rmse_ind_ref # Per parameter importance
     
        res_row = [  fi_index_list[n][0],  fi_index_list[n][-1],  tot_fi ] + list(param_fi)
        if n==0:
            fi_table = Table(np.array(res_row).reshape(1,-1), names=col_names)
        else:
            fi_table.add_row( res_row )
            ascii.write(fi_table, fi_filename, format='commented_header', delimiter='\t', overwrite=True)
        
        
        if time()-t_mid > 15*60:
            t_mid = time()
            print("Time passed: %.2f min (%.2f h)"%( (t_mid - t_start)/60, (t_mid-t_start)/3600) )
            
    t_end = time()
    ascii.write(fi_table, fi_filename, format='commented_header', delimiter='\t', overwrite=True)
    dt = t_end - t_start
    print('Time taken: %.2f min (%.2f hour)'%( dt/60, dt/3600) )
    
