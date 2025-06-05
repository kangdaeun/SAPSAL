#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 13:35:59 2025

@author: daeun

Make D(c) distribution for multiple networks sharing astro, real DB : hps
save in one directory : plot_dir
"""

import numpy as np
import matplotlib.pyplot as plt
import os,sys
# from time  import time
# import matplotlib.colors as clr
# import copy
# import matplotlib.cm as cm
import torch
# import matplotlib.ticker as ticker
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from pathlib import Path
# from astropy.table import Table, join, vstack
# import matplotlib.gridspec as gridspec
# import pandas as pd


from sys import platform
if platform == "linux" or platform == "linux2":
    main_dir = '/home/PERSONALE/daeun.kang/kde_ar/cinn_ssp/'
#     # linux
elif platform == "darwin":
    main_dir = '/Users/daeun/Astronomy/Pj_cinn_ssp/'
#     # OS X
# elif platform == "win32":
#     # Windows...
print("main_dir",main_dir)
sys.path.append(main_dir+"cINN_SSP/")
# sys.path.append("/Users/daeun/Astronomy/Pj_cinn_ssp/Networks/")
from sapsal.cINN_config import read_config_from_file
from sapsal.data_loader import DataLoader


config_dic={}
# config_dic['test']={'config_file':(main_dir+'Factory/SpD/DA_TGARsL_mMUSE/', 'test4/c_test42.py')}
# for i in range(1,7):
#     key = 'N%d'%i
#     dirpath = main_dir+'Factory/SpD/DA_TGARsL_mMUSE/'
#     filepath = 'test4/c_test4%d.py'%i
#     config_dic[key] = {'config_file': (dirpath, filepath)}
for i in range(10):
    key = 'N%d'%i
    dirpath = main_dir+'Factory/SpD/DA_TGARsL_mMUSE/'
    filepath = 'dhps2/configs/c_SpD_DA_TGARsL_mMUSE_%02d.py'%i
    config_dic[key] = {'config_file': (dirpath, filepath)}
    
plot_dir = main_dir+'Factory/SpD/DA_TGARsL_mMUSE/dhps2/plots/Ddistr/'
if not os.path.exists(plot_dir):
    os.system('mkdir -p %s'%plot_dir)

key_list = list(config_dic.keys())

for key, ddic in config_dic.items():
    cINN_dir, config_file = ddic['config_file']
    c = read_config_from_file(cINN_dir+config_file, proj_dir=cINN_dir)
    if platform == "darwin":
        c.device = 'cpu'
    config_dic[key]['config']=c
    network_name = os.path.basename(c.filename).replace('.pt','')
    config_dic[key]['network_name'] = network_name
    try:
        a=int(network_name.split('_')[-1])
        net_code = '_'.join(network_name.split('_')[:-1])
    except:
        net_code = network_name
    config_dic[key]['net_code'] = net_code
    print(key)
    print("Network name: %s"%network_name)
    print("Net code: %s"%net_code)
    # c.print_short_setting()

mkey = key_list[0]
c = config_dic[mkey]['config']
exp = c.import_expander()

# Read Train, Test data and real data used in domain adaption
# currenly all networks in config_dic use the same astro setup and the same real data
# only read for the 1st config
for key, ddic in config_dic.items():
    c=ddic['config']
    if key == key_list[0]:
        astro = DataLoader(c)
        veil_flux = False
        extinct_flux = False
        if astro.random_parameters is not None:
            if "veil_r" in astro.random_parameters.keys():
                veil_flux = True
            if "A_V" in astro.random_parameters.keys():
                extinct_flux = True
                
        test_set, train_set = astro.get_splitted_set(rawval=True, smoothing = False, smoothing_sigma = None,
                                                     normalize_flux=c.normalize_flux, 
                                                     normalize_total_flux=c.normalize_total_flux, 
                                                     normalize_mean_flux=c.normalize_mean_flux,
                                                     veil_flux = veil_flux, extinct_flux = extinct_flux, 
                                                     random_seed=0,
                                                     )
        ddic['train_set'] = train_set
        ddic['test_set'] = test_set
        ddic['obs_real'] = astro.get_real_data(rawval=True, normalize_flux=c.normalize_flux, 
                                                     normalize_total_flux=c.normalize_total_flux, 
                                                     normalize_mean_flux=c.normalize_mean_flux,
                                              )
        
# obs -> y -> feature -> D
obs_test = config_dic[key_list[0]]['test_set'][1]
obs_train = config_dic[key_list[0]]['train_set'][1]
obs_real = config_dic[key_list[0]]['obs_real']

device = 'mps'

for key, ddic in config_dic.items():
    # if key==key_list[1]:
    c = ddic['config']
    c.device = device
    c.load_network_model()
    c.network_model.eval()

    with torch.no_grad():
        # obs -> y
        y = torch.Tensor(c.obs_to_y(obs_train)).to(c.device)
        features = c.network_model.cond_net.features(y)
        D_train = c.network_model.da_disc(features)

        y = torch.Tensor(c.obs_to_y(obs_test)).to(c.device)
        features_test = c.network_model.cond_net.features(y)
        D_test = c.network_model.da_disc(features_test)
    
        yr = torch.Tensor(c.obs_to_y(obs_real)).to(c.device)
        features_real = c.network_model.cond_net.features(yr)
        D_real = c.network_model.da_disc(features_real)

        loss_train = torch.nn.functional.binary_cross_entropy_with_logits(D_train, torch.zeros_like(D_train)).data.cpu().numpy()
        loss_test = torch.nn.functional.binary_cross_entropy_with_logits(D_test, torch.zeros_like(D_test)).data.cpu().numpy()
        loss_real = torch.nn.functional.binary_cross_entropy_with_logits(D_real, torch.ones_like(D_real)).data.cpu().numpy()

        D_train= torch.sigmoid(D_train).data.cpu().numpy()
        D_real = torch.sigmoid(D_real).data.cpu().numpy()
        D_test = torch.sigmoid(D_test).data.cpu().numpy()

    ddic['f_train'] = features.data.cpu().numpy()
    ddic['f_test'] = features_test.data.cpu().numpy()
    ddic['f_real'] = features_real.data.cpu().numpy()
    
    ddic['D_train'] = D_train.ravel()
    ddic['D_test'] = D_test.ravel()
    ddic['D_real'] = D_real.ravel()
    
    ddic['loss_train'] = loss_train.ravel()
    ddic['loss_test'] = loss_test.ravel()
    ddic['loss_real'] = loss_real.ravel()
    

# ddic = config_dic[key_list[3]]
for key, ddic in config_dic.items():
        
    D_train = ddic['D_train']
    D_test = ddic['D_train']
    D_real = ddic['D_real']
    
    loss_train = ddic['loss_train']
    loss_test = ddic['loss_test']
    loss_real = ddic['loss_real']

    fig, ax = plt.subplots(1,1, figsize=[5, 4.], tight_layout=1)
    kwarg = {'density':True, 'bins':100, 'alpha':0.5}
    for val, label in zip([D_train.ravel(), D_test.ravel(), D_real.ravel()], ['Dtrain','Dtest','Dreal']):
        avg = np.mean(val); std=np.std(val)
        txt = label + ': %.1f$\pm$%.1f'%(avg, std)
        _ = ax.hist(val, label=txt, **kwarg)
    ax.legend()
    ax.set(ylabel='Probability density', xlabel='Discriminator(condition)', title=ddic['network_name'])
    figname=f"Ddistr_{ddic['network_name']}"
    
    
    
    
    # fig, ax = plt.subplots(figsize=[6,5], tight_layout=1)
    # ax=axis[1]
    # kwarg = {'density':True, 'bins':100, 'alpha':0.5}
    # for val, label in zip([loss_train.ravel(), loss_test.ravel(), loss_real.ravel()], ['train','test','real']):
    #     if label=='Dreal': continue
    #     avg = np.mean(val); std=np.std(val)
    #     txt = label + ': %.1f$\pm$%.1f'%(avg, std)
    #     _ = ax.hist(val, label=txt, **kwarg)
    
    
    # ax.legend()
    # fig.suptitle(key)
    # figname=f'Ddistr_{key}'
    # plot_dir = 
