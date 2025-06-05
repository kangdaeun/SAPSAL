#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 12:24:35 2023

@author: daeun
"""

import numpy as np
import sys
import os
import torch

from argparse import ArgumentParser

# sys.path.append("/export/scratch/dekang/ECOGAL/cinn_ssp/Networks/")

import sapsal.tools.hs_tools as hs_tools
from sapsal.tools.test_tools import plot_z
from sapsal.cINN_config import read_config_from_file
from sapsal.data_loader import DataLoader
from sapsal.execute import train_ftrans_network

NUM_CONFIG = 70 # number of config file for one random seed
MAX_EPOCH = 250 # max epoch for each training. usually its hard to meet converge 
SEED = 15



SEARCH_PARAMETERS = {
    "gamma" : (hs_tools.rand_in, {"a":0.1, "b":0.8}),
    "adam_betas" : (hs_tools.rand_in_discrete, {"options":[(0.8, 0.8), (0.9, 0.9)]}),
    "lr_init" : (hs_tools.rand_in_log10, {"a":-4, "b":-2}),
    "l2_weight_reg" : (hs_tools.rand_in_log10, {"a":-4.3, "b":-2}),
    "meta_epoch" : (np.random.randint, {"low":5, "high":16}),
    "n_blocks" : (np.random.randint, {"low":1, "high":8+1}),
    "internal_layer": (np.random.randint, {"low":3, "high":9+1}),
    "internal_width" : (hs_tools.rand_in_log2, {"a":8 ,"b":9}), # usually fixed to 256
    # "batch_size" : (tools.rand_in_log2, {"a":7, "b":10}),
    # "n_its_per_epoch" : (tools.rand_in_log2, {"a":8, "b":10}),
    "seed_weight_init" : (np.random.randint, {"low":1, "high":241542})
    }


# Parse optional command line arguments
parser = ArgumentParser()
# parser.add_argument("-c", "--config", dest="config_file", default='./config.py',
#                     help="Run with specified config file as basis.")

parser.add_argument('config_file', help="Run for this cINN config.")
parser.add_argument('-s','--suffix', required=False, default=None, help="Output suffix")
parser.add_argument('-o','--outputdir', required=False, default=None, help="Output directory to save suffix/ directory")
parser.add_argument('-r','--renew', required=False, default=False, help="Renew the saved network")

args = parser.parse_args()

# Import default config
config_file = args.config_file
c = read_config_from_file(config_file, verbose=False)
network_name = os.path.basename(c.filename).replace('.pt','')


if args.suffix is not None:
    suffix = str(args.suffix)
else:
    suffix = network_name

if args.outputdir is not None:
    outdir = str(args.outputdir)
else:
    outdir = os.path.dirname(c.filename)+'/FTNet/'
print("Save here: %s"%outdir)

renew=False
if args.renew is not None:
    if args.renew == 'False' or args.renew =='0':
        renew = False
    if args.renew =='True' or args.renew=='1':
        renew = True
else:
    renew = False



# Prepare final filename and save directory
ftnet_filename = outdir+'model_FTr_%s/FTr_%s.pt'%(suffix, suffix)
ft_configfile = outdir+'c_FTr_%s.py'%suffix
print("FTtransformNet name: %s"%ftnet_filename)
print("FTNet config name: %s"%ft_configfile)

if os.path.exists(ftnet_filename) and os.path.exists(ft_configfile):
    if renew:
        print("Delete and retrain the network")
        os.system("rm -rf %s"%ftnet_filename)
        os.system("rm -rf %s"%ft_configfile)

    else:
        sys.exit("Already trained the network")

# Load training and test data. And transform to feature
exp = c.import_expander()
astro = DataLoader(c)
device = exp.find_gpu_available(gpu_max_memory=0.2)
c.device =  device
c.load_network_model()
# obs_all -> y_all -> feature_all 

veil_flux = False
extinct_flux = False
if astro.random_parameters is not None:
    if "veil_r" in astro.random_parameters.keys():
        veil_flux = True
    if "A_V" in astro.random_parameters.keys():
        extinct_flux = True
test, train = astro.get_splitted_set(rawval=True, smoothing = False, smoothing_sigma = None,
                                             normalize_flux=c.normalize_flux, 
                                             normalize_total_flux=c.normalize_total_flux, 
                                             normalize_mean_flux=c.normalize_mean_flux,
                                             veil_flux = veil_flux, extinct_flux = extinct_flux, 
                                             random_seed=0,
                                     )
y_test = c.obs_to_y(test[1])
y_train = c.obs_to_y(train[1])

if c.prenoise_training:
    # also need sigma --> Use minimum value and do not perturb y
    sig_test =  np.zeros(shape=y_test.shape) + c.lsig_min
    y_test = np.hstack([y_test, astro.unc_to_sig(10**sig_test)])
    
    sig_train =  np.zeros(shape=y_train.shape) + c.lsig_min
    y_train = np.hstack([y_train, astro.unc_to_sig(10**sig_train)])

f_test = c.network_model.cond_net.features(torch.Tensor(y_test).to(c.device)).data.cpu().numpy()
f_train = c.network_model.cond_net.features(torch.Tensor(y_train).to(c.device)).data.cpu().numpy()

    



print("Train network")

if not os.path.exists(outdir):
    os.system("mkdir -p %s"%outdir)

# Prepare config for FTransformNet
c_ft = c.copy()
c_ft.config_file = ft_configfile
c_ft.model_code = 'FTransformNet_GLOW'
c_ft.filename = ftnet_filename
c_ft.batch_size = 1024 # fixed batch size for the speed

np.random.seed(SEED)

train_status = 0

c_list = []
for i in range(NUM_CONFIG+1):
    c_list.append( hs_tools.randomize_config(c_ft, SEARCH_PARAMETERS,
                                    c_ft.filename, c.device, 10000,
                                    adjust_gamma_n_epoch=False) )

num = 0

while train_status!=1 and num < NUM_CONFIG:
    
    c_ft = c_list[num]
    num+=1
    print(num)
    
    c_ft.filename = ftnet_filename
    
    
    # Train
    model, train_status = train_ftrans_network(c=c_ft, feature_test=f_test, feature_train=f_train, 
                  verbose=True, max_epoch=MAX_EPOCH, return_model=True, return_training_status=True)
    
    # z evaluation
    with torch.no_grad():
        output, jac = model( torch.Tensor(f_test).to(device)   )
    z_all = output.data.cpu().numpy()


    corr = np.cov(z_all, rowvar=False)
    # non diagnal abs < 0.1
    roi_ndiag = ~np.eye(corr.shape[0],dtype=bool)
    check = np.sum(abs(corr[roi_ndiag]) > 0.15)/2 > 0.05*c.y_dim_features
    if check:
        train_status = -1
        print("Covariance fail")
        
    bins = 100; hrange=(-7,7)
    stdnormal = lambda a: np.exp(-0.5*a*a)/np.sqrt(2*np.pi)
    max_resi = []
    resi_ind = []
    rmse_ind = []
    yhis_list = []
    for i in range(z_all.shape[1]+1):
        if i == 0:
            yhis, xhis = np.histogram(z_all.ravel(), bins=bins, range=hrange, density=True)
        else:
            yhis, xhis = np.histogram(z_all[:,i-1], bins=bins, range=hrange, density=True)
            yhis_list.append(yhis)
        xp = 0.5*(xhis[:-1]+xhis[1:])
        resi = yhis - stdnormal(xp)
        if i>0:
            resi_ind.append(resi)
            rmse_ind.append( np.sqrt(np.nanmean(resi**2.) ))
        max_resi.append(np.nanmax(abs(resi)))
    max_resi = np.array(max_resi)
    rmse_ind = np.array(rmse_ind)
    rmse = np.sqrt(np.sum(rmse_ind**2))
    # print(np.sqrt(np.sum(rmse_ind**2)))
    
    if np.sum(max_resi > 0.15) > 0:
        train_status = -1
        print("Z residual fail")
        
    figurename = c_ft.filename+ '_z_cov_pdf.pdf'
    r1=plot_z(z_all, figname=figurename, corrlabel=False, legend=False, covariance=True, color_letter='r')
    
    if train_status != -1:
        str_x_names, str_y_names = hs_tools.find_str_names(c, c.config_file, dim_max=20)
        c_ft.save_config(config_file = ft_configfile,
                         config_comment = "random selection, coverged",
                          str_x_names = str_x_names, str_y_names=str_y_names, verbose=False )
        print("Train finished. Saved config")
        train_status=1
        
    elif num==NUM_CONFIG:
        print("  tried %d times but could not find converged config"%(NUM_CONFIG))

        
