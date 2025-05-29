#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:33:24 2022

@author: daeun

Evaluation code 
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
import os
import sys
import matplotlib.cm as cm
# from pathlib import Path
from time import time
import pickle
import torch
from argparse import ArgumentParser

# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patheffects as PathEffects

# import matplotlib.colors as clr


import matplotlib.ticker as ticker

# import torch
from astropy.table import Table, vstack

from cINN_set.cINN_config import read_config_from_file
from cINN_set.data_loader import DataLoader
from cINN_set.tools import test_tools as tools
from cINN_set.tools.logger import Logger
# from cINN_set.tools.test_tools import plot_calibration
import copy


"""
eval_TVP:
- all posteriors + MAP 
- TvP histogram + TvMAP histogram info
- two figures
- save MAP values

calib:
- calibaration error per parm as a function of confidence level
- uncertainty 
- calib table file + figure

RMSE
- RMSE using all post 
- RMSE using MAPs
"""


##########
## Setup ##
###########

GROUP_SIZE = 400 # n(obs) per one iteration of get_posterior
N_PRED = 4096 # latent variable sampling  


GPU_MAX_LOAD = 0.4           # Maximum compute load of GPU allowed in automated selection
GPU_MAX_MEMORY = 0.4         # Maximum memory load of GPU allowed in automated selection
GPU_WAIT_S = 600             # Time (s) to wait between tries to find a free GPU if none was found
GPU_ATTEMPTS = 10            # Number of times to retry finding a GPU if none was found
GPU_EXCLUDE_IDS = [1,2] # List of GPU IDs that are to be ignored when trying to find a free GPU, leave as empty list if none

VERBOSE = True           # Switch for progress messages

LOG_MODE = True     # Print both console and log # this does not set to save log!




##########
## MAIN ##
##########

# if __name__=='__main__':
#     config_file = sys.argv[1]

def evaluate(c, astro=None, lsig_fix=None, verbose=VERBOSE):
    
    eval_TvP = True
    eval_calib = True
    eval_RMSE = True
    eval_z = True
    
    run_forward = False
    run_all_post = False
    run_all_map = False
    run_ind_post = False
    run_calib = False
    run_hist = False
    run_hist_map = False
    run_Ddist = False
    
    if c.domain_adaptation: eval_Ddist = True  
    else: eval_Ddist=False
    
    # Setup file and figure names
    
    filename_dic = {}; figurename_dic = {}
    if c.prenoise_training==True and lsig_fix is not None:
        # change filenames _{}
        adding = "_lsig{:.4g}".format(lsig_fix)
    
    if eval_z:
        print("Request eval_z")
        figurename_dic['z_cov'] = c.filename+ tools.testfigure_suffix['z_cov']
        figurename_dic['z_corr'] = c.filename+tools.testfigure_suffix['z_corr']
        
        filename_dic['z_test'] = c.filename+tools.testfile_suffix['z_test']
        figurename_dic['z_qq'] = c.filename+tools.testfigure_suffix['z_qq']
        
        for file in [figurename_dic['z_cov'], figurename_dic['z_corr'], filename_dic['z_test'], figurename_dic['z_qq']]:
            if not os.path.exists(file):
                run_forward = True
    
    if eval_TvP:
        print("Request eval_TvP")
        filename_dic['TvP'] = c.filename + tools.testfile_suffix['TvP'] # all
        filename_dic['MAP'] = c.filename + tools.testfile_suffix['MAP'] # map -> all
        filename_dic['RMSE'] = c.filename + tools.testfile_suffix['RMSE'] # all and map
        
        # for Specific lsig for Noise-Net
        if c.prenoise_training==True and lsig_fix is not None:
            # change filenames _{}
            filename_dic['TvP'] = filename_dic['TvP'][:-4] + adding + filename_dic['TvP'][-4:]
            filename_dic['MAP'] = filename_dic['MAP'][:-4] + adding + filename_dic['MAP'][-4:]
            filename_dic['RMSE'] = filename_dic['RMSE'][:-4] + adding + filename_dic['RMSE'][-4:]
        
        if os.path.exists(filename_dic['TvP']):
            print("\tRead histogram data")
            with open(filename_dic['TvP'], 'rb') as hf:
                dic = pickle.load(hf)
                hist_dic = dic['hist_dic']
                map_hist_dic = dic['map_hist_dic']
        else:
            run_all_post = True
            run_hist = True
            run_hist_map = True
            
        if os.path.exists( filename_dic['MAP'] ):
            print("\tRead MAP data")
            _ = ascii.read(filename_dic['MAP'], format='commented_header', delimiter='\t')
            map_list = np.array(_[c.x_names]).view(float).reshape(-1, len(c.x_names))
        else:
            run_all_map = True
        
        if os.path.exists( filename_dic['RMSE'] ):
            print("\tRead RMSE data")
            rmse_table = ascii.read(filename_dic['RMSE'], format='commented_header', delimiter='\t')
        else:
            eval_RMSE = True
            run_all_post = True
        
        # check figures
        figurename_dic['TvP'] = c.filename + tools.testfigure_suffix['TvP']
        figurename_dic['TvP_MAP'] = c.filename + tools.testfigure_suffix['TvP_MAP']
        
        # for Specific lsig for Noise-Net
        if c.prenoise_training==True and lsig_fix is not None:
            # change filenames _{}
            figurename_dic['TvP'] = figurename_dic['TvP'][:-4] + adding + figurename_dic['TvP'][-4:]
            figurename_dic['TvP_MAP'] = figurename_dic['TvP_MAP'][:-4] + adding + figurename_dic['TvP_MAP'][-4:]       
        
        
        if (run_all_post == False) * (run_all_map == False):
            if os.path.exists( figurename_dic['TvP']) * os.path.exists( figurename_dic['TvP']):
                print('\tFigures and data already exist. Do not need eval_TvP.')
                eval_TvP = False
        
        
    if eval_RMSE:
        print("Request eval_RMSE")
        
        filename_dic['RMSE'] = c.filename + tools.testfile_suffix['RMSE'] # all and map
        filename_dic['MAP'] = c.filename + tools.testfile_suffix['MAP'] # map -> all
        
        # for Specific lsig for Noise-Net
        if c.prenoise_training==True and lsig_fix is not None:
            # change filenames _{}
            filename_dic['MAP'] = filename_dic['MAP'][:-4] + adding + filename_dic['MAP'][-4:]
            filename_dic['RMSE'] = filename_dic['RMSE'][:-4] + adding + filename_dic['RMSE'][-4:]
            
        
        if os.path.exists( filename_dic['MAP'] ):
            print("\tRead MAP data")
            _ = ascii.read(filename_dic['MAP'], format='commented_header', delimiter='\t')
            map_list = np.array(_[c.x_names]).view(float).reshape(-1, len(c.x_names))
            
            if os.path.exists( filename_dic['RMSE'] ):
                print("\tRMSE and MAP files already exist. Do not need eval_RMSE.")
                eval_RMSE = False
            else:
                run_all_post = True
        else:
            run_all_map = True
            run_all_post = True
            run_ind_post = True
            
    if eval_calib:
        print("Request eval_calib")
        
        filename_dic['calib'] = c.filename + tools.testfile_suffix['calib'] # all
        figurename_dic['calib'] = c.filename + tools.testfigure_suffix['calib']
        
        # for Specific lsig for Noise-Net
        if c.prenoise_training==True and lsig_fix is not None:
            # change filenames _{}
            filename_dic['calib'] = filename_dic['calib'][:-4] + adding + filename_dic['calib'][-4:]
            figurename_dic['calib'] = figurename_dic['calib'][:-4] + adding + figurename_dic['calib'][-4:]
        
        
        if os.path.exists(filename_dic['calib']):
            print("\tRead calib data") 
            calib_table = ascii.read(filename_dic['calib'], delimiter='\t', format='commented_header')
            if os.path.exists(figurename_dic['calib'] ):
                print("\tCalib data and figure already exist. Do not need eval_calib.")
                eval_calib = False
        else:
            run_all_post = True
            run_ind_post = True
            run_calib = True
            
    if eval_Ddist: # only for domain adaptaion
        print("Request eval_Ddist")
        figurename_dic['Ddist'] = c.filename + tools.testfigure_suffix['Ddist']
        if os.path.exists(figurename_dic['Ddist']):
            run_Ddist = False
        else:
            run_Ddist = True
        
        
    if run_all_map:
        run_ind_post = True
        run_all_post = True
    
    print()
    if run_forward:
        print('Need to run forward for latent tests')
    if run_Ddist:
        print('Need to run Ddist for domain adapation test')
    if run_all_post:
        print('Need to run posteriors')
    if run_all_map:
        print('Need to calculate MAPs')
    if run_calib:
        print('Need to calculate calibrations')
    if run_hist:
        print('Need to calculate TvP histogram')
    if run_hist_map:
        print('Need to calculate TvP MAP histogram')
    print()  
    
    network_name = os.path.basename(c.filename).replace('.pt','')
    
    #%% Load data 
    # astro = DataLoader(c)
    if astro is None:
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
    param_test = test_set[0]; obs_test = test_set[1]   
    N_test = len(param_test)
    
    #%% 1) Latent tests
    if run_forward:
        c.load_network_model()
        # for z tests, keep similar to training situation: raundom sigma, smoothing 
        z_all = tools.calculate_z(c.network_model, astro, smoothing=c.train_smoothing)
        tools.plot_z(z_all, figname=figurename_dic['z_cov'], corrlabel=True, title=network_name,
                     legend=True, covariance=True, cmap=cm.get_cmap("gnuplot"), color_letter='r', return_figure=False)#, yrange1=[-0.04, 0.6], yrange2=[-0.1, 0.2])
        tools.plot_z(z_all, figname=figurename_dic['z_corr'], corrlabel=True, title=network_name,
                     legend=True, covariance=False, cmap=cm.get_cmap("gnuplot"), color_letter='r', return_figure=False)
        if VERBOSE:
            print("Saved Z covariance and distributions")
        df = tools.latent_normality_tests(z_all, filename=filename_dic['z_test'])
        if VERBOSE:
            print("Saved Z normality tests")
        q1 = tools.qq_plot(z_all, figname=figurename_dic['z_qq'], title=network_name)
        if VERBOSE:
            print("Saved Z q-q plot")
    
    #%% Domain adaptaion test
    if run_Ddist:
        c.load_network_model()
        D_test, D_real = tools.calculate_D(c.network_model, astro, smoothing=c.train_smoothing)
        tools.plot_D_distribution(c, D_test, D_real, figname=figurename_dic['Ddist'], return_figure=False,
                            title=network_name, titlesize='large'  )
        if VERBOSE:
            print("Saved Domain adaptaion test: D(c) distributions")
    
    #%% 2) Run all: calib. RMSE, Histograms 
    # Setup general parameters for p(x) sampling
    # GROUP_SIZE: # n(obs) per one iteration
    n_group = np.ceil(len(param_test)/GROUP_SIZE).astype(int)
    
    
    # eval_TvP:
    hist_range_dic = {}
    for i_param, param in enumerate(astro.x_names):
        xrange = [np.min(test_set[0][:,i_param]), np.max(test_set[0][:,i_param])]
        yrange = xrange
        hist_range_dic[param] = (xrange, yrange)
    
    # Setting for histogram
    nbin=100 # histogram bins for True vs All
    if len(obs_test) < 1e4:
        nbin = np.ceil(np.sqrt(len(obs_test))).astype(int)
    
    def determine_hist_bins(param):
        # add condition for discretized parameters
        # if param=='N_cluster':
        #     bins=[np.arange(0.5, np.max(test_table[param].data)+0.51, 1.), np.arange(0.5, np.max(test_table[param].data)+0.51, 1.)]
        # elif param=='phase':
        #     bins=[np.arange(-0.5, 3.51, 1.), np.arange(-0.5, 3.51, 1.)]
        # else:
        #     bins = nbin
        if param=='library':
            bins=[np.arange(-0.5,1.51,1),np.arange(-0.5,1.51,1)]
        else:
            bins = nbin
        return bins
    
    
    # special parameters
    discretized_parameters = ['library']
    
    # 2) map keywords
    map_kwarg = {'bw_method':'silverman', 
                 'n_grid':1024, 
                 'use_percentile': None,
                 'plot':False }
    

        
    
    # Calibration setting
    # how many different confidences to look at
    n_steps = 100
    q_values = []
    confidences = np.linspace(0., 1., n_steps+1, endpoint=True)[1:] # n_steps: 0.01, 0.02, ..., 0.99, 1.00
    for conf in confidences:
        q_low  = 0.5 * (1 - conf)
        q_high = 0.5 * (1 + conf)
        q_values += [q_low, q_high] #-> q_values: 2 * n_steps
        
    # result: calib_table
        
    if run_all_post:
        
        print("Start process")
        t_start = time()
        
        if run_all_map:
            map_list = []
            
        if run_hist:
            hist_dic = {}
        
        if eval_RMSE:
            sum_dev_param = 0
            N_dev = 0
            sum_dev_x = 0
    
        for i_group in range(n_group):
            obs_group = obs_test[i_group*GROUP_SIZE:(i_group+1)*GROUP_SIZE]
            param_group = param_test[i_group*GROUP_SIZE:(i_group+1)*GROUP_SIZE]
            
            # prenoise : use middle value for p(sigma). one sigma for all y components
            if c.prenoise_training==True:
                # if c.unc_corrl == 'Ind_Man' or c.unc_corrl == 'Ind_Unif' or c.unc_corrl == 'Single' :
                if lsig_fix is not None: # for requested lsig
                    lsig_mid = lsig_fix
                else:
                    # Currently not supported for Seg_Flux option
                    if c.unc_sampling == 'gaussian':
                        lsig_mid = c.lsig_mean
                    elif c.unc_sampling == 'uniform':
                        lsig_mid = 0.5*(c.lsig_min + c.lsig_max)
                
                if np.ndim(lsig_mid)==0: # one value
                    lsig_mid = np.repeat(lsig_mid, obs_group.shape[1]) # const -> 1D array
                
                # same error for all observations
                unc_group = 10**np.repeat([lsig_mid], obs_group.shape[0], axis=0)
            else:
                unc_group = None
                
            if c.use_flag == True:
                flag_group = astro.create_random_flag(obs_group.shape[0])
            else:
                flag_group = None,
            
            # posterior per group
            post_list = astro.exp.get_posterior(obs_group, c, N=N_PRED, use_group=True, group=GROUP_SIZE, 
                                                unc=unc_group, flag=flag_group,
                                          return_llike=False, quiet=True)
            
            
            # RMSE for all post
            if eval_RMSE:
                sum_dev_param_group = (post_list - param_group.reshape(-1, 1, len(c.x_names)).repeat(N_PRED,axis=1) ).reshape(-1, len(c.x_names))**2. # (model x N_PRED) x x_dim
                sum_dev_x_group = ( c.params_to_x(post_list) - c.params_to_x(param_group).reshape(-1, 1, len(c.x_names)).repeat(N_PRED,axis=1)).reshape(-1, len(c.x_names))**2.
                
                if np.sum( np.isfinite(sum_dev_param_group)==False ) >0:
                    roi = np.array([True]*len(sum_dev_param_group))
                    for i_param, param in enumerate(c.x_names):
                        roi *= (np.isfinite(sum_dev_param_group[:, i_param]))
                    
                    sum_dev_param += np.sum( sum_dev_param_group[roi], axis=0)
                    sum_dev_x += np.sum( sum_dev_x_group[roi], axis=0)
                    N_dev += np.sum(roi)
                    
                else:
                    sum_dev_param += np.sum( sum_dev_param_group, axis=0)
                    sum_dev_x += np.sum( sum_dev_x_group, axis=0)
                    N_dev += sum_dev_param_group.shape[0]
            
            
            # individual models: MAP, calib
            
            # calib
            if run_calib:
                uncert_intervals_group = np.zeros(shape=(len(post_list), n_steps, len(c.x_names) ) ) # len(test_set) x 100 x x_dim x 
                inliers_group = uncert_intervals_group.copy()
                xs_group = c.params_to_x(param_group)
            
            if run_ind_post:
                for i_model, post in enumerate(post_list):
    
                    if run_all_map:
                        map_list.append( astro.exp.calculate_map(post,c, **map_kwarg ) )
    
                    # calib with x-scaled
                    if run_calib:
                        x_margins_2d = np.quantile(c.params_to_x(post), q_values, axis=0)
                        for i_param, param in enumerate(c.x_names):
                            x_margins = list(x_margins_2d[:, i_param])
    
                            for i_step in range(n_steps):
                                x_low, x_high = x_margins.pop(0), x_margins.pop(0)
                                uncert_intervals_group[i_model, i_step, i_param] = x_high - x_low
                                inliers_group[i_model, i_step, i_param] = int(xs_group[i_model][i_param] < x_high and xs_group[i_model][i_param] > x_low) 
                            # check target x is in within (xlow, x_high) -> True/False -> 1/0
    
    
               
            # add group result of calib
            if run_calib:
                if i_group==0:
                    uncert_intervals = uncert_intervals_group
                    inliers = inliers_group
                else:
                    uncert_intervals = np.append(uncert_intervals, uncert_intervals_group, axis=0)
                    inliers = np.append(inliers, inliers_group, axis=0)
                
                
            if run_hist:
                # calculate histogram (True vs All) for this group
                for i_param, param in enumerate(c.x_names):
                    true1d = param_group[:,i_param].reshape(-1,1).repeat(N_PRED,axis=1).ravel()
                    post1d = post_list.reshape(-1, len(c.x_names))[:,i_param]
    
                    bins = determine_hist_bins(param)
    
                    if i_group==0:
                        xrange, yrange = hist_range_dic[param]
                        H, xedges, yedges = np.histogram2d(true1d, post1d, bins=bins, range=(xrange,yrange))
                        hist_dic[param] = {'H': H,'xedges':xedges, 'yedges': yedges  }
    
                    else:
                        H, _1, _2 = np.histogram2d(true1d, post1d, bins=(hist_dic[param]['xedges'], hist_dic[param]['yedges']) )
                        hist_dic[param]['H'] += H
                        
            
             
            if (i_group%10==0):
                print('{:d}th group (group={:d}): {:.2f} min'.format(i_group, GROUP_SIZE, (time()-t_start)/60. ))
        # end for 
        
        # MAP 
        if run_all_map:
            map_list = np.array(map_list)
        
        # calib
        if run_calib:
            inliers = np.mean(inliers, axis=0) # 100 x 7
            calib_err = inliers - confidences.reshape(-1,1)
            uncert_intervals = np.median(uncert_intervals, axis=0)
            
            calib_table = Table(confidences.reshape(-1,1), names=['confidence'])
            for i, param in enumerate(astro.x_names):
                calib_table[param+'_clb_err'] = calib_err[:,i]
            for i, param in enumerate(astro.x_names):
                calib_table[param+'_unc_intrv'] = uncert_intervals[:,i]
    
        
        # TvP: MAP histogram (after for loop)
        if run_hist_map:
            map_hist_dic = {}
            for i_param, param in enumerate(c.x_names):
                xtrue = param_test[:,i_param]
                map_values = map_list[:,i_param]
                xrange, yrange = hist_range_dic[param]
    
                bins = determine_hist_bins(param)
    
                H, xedges, yedges = np.histogram2d(xtrue, map_values,
                                                  bins=bins, range=(xrange,yrange))
                map_hist_dic[param] = {'H': H,'xedges':xedges, 'yedges': yedges  }
    
            
        # end 
        t_end= time()
        print('Time taken: {:.2f} min'.format((t_end-t_start)/60. ))
    
        
        
    #%% eval remainings
    if eval_RMSE:
        
        # save MAP information
        if run_all_map:
            ascii.write(Table(map_list, names=c.x_names), filename_dic['MAP'], format='commented_header', delimiter='\t', overwrite=True)
        
        # if there is any preprocessing in MAP... preprocess true value as well (param_test)
        
        ptrue = param_test.copy()
        xtrue = c.params_to_x(ptrue)
        
        # RMSE using MAP
        dparam = map_list- ptrue
        dx = c.params_to_x(map_list) - xtrue
        
        rmse_p = []
        rmse_x = []
        for i, param in enumerate(c.x_names):
            roi = np.isfinite(map_list[:, i])
            if np.sum(roi) < len(map_list):
                print('{}: {:d} nan cases for {:d} test models ({:.1f}%)'.format(param, len(map_list)-np.sum(roi), 
                                                                                 len(map_list), 
                                                                                 100*(len(map_list)-np.sum(roi))/len(map_list)  )  )
            rmse_p.append( np.sqrt( np.sum(dparam[roi, i]**2)/np.sum(roi) )  )
            rmse_x.append( np.sqrt( np.sum(dx[roi, i]**2)/np.sum(roi) )  )
        rmse_p = np.array(rmse_p)
        rmse_x = np.array(rmse_x)
        
        rmse_table = Table( np.vstack( (rmse_p, rmse_x) ), names=c.x_names)
        
        # RMSE using all post
        rmse_p = np.sqrt( sum_dev_param / N_dev)
        rmse_x = np.sqrt( sum_dev_x / N_dev )
        rmse_table = vstack([rmse_table, Table( np.vstack( (rmse_p, rmse_x) ), names=c.x_names)], join_type='exact')
        
        code_names = ['RMSE_MAP_PARAM', 'RMSE_MAP_X', 'RMSE_ALL_PARAM', 'RMSE_ALL_X']
        rmse_table['type'] = code_names
        rmse_table = rmse_table[ ['type']+c.x_names]
        
        # save rmse information
        ascii.write(rmse_table, filename_dic['RMSE'], format='commented_header', delimiter='\t', overwrite=True)
        print("Saved RMSE file")
        
        # save MAP information
        if run_all_map:
            ascii.write(Table(map_list, names=c.x_names), filename_dic['MAP'], format='commented_header', delimiter='\t', overwrite=True)
            print("Saved MAP file")    
        
    

    if eval_TvP:
        
        """
        File to save
        - histogram info of True Vs All
        - histogram info of True vs MAP
        """
        # Save histogram 
        if (run_hist==True) or (run_hist_map==True) :
            s_f = open(filename_dic['TvP'], 'wb')
            pickle.dump({'hist_dic':hist_dic, 'map_hist_dic':map_hist_dic}, s_f)
            s_f.close()
            print('Saved histogram file')
        
        # save MAP information
        if run_all_map:
            ascii.write(Table(map_list, names=c.x_names), filename_dic['MAP'], format='commented_header', delimiter='\t', overwrite=True)
            print("Saved MAP file")
        
    if eval_calib:
        
        if not os.path.exists(filename_dic['calib']):
            ascii.write(calib_table, filename_dic['calib'], delimiter='\t', format='commented_header')
            print("Saved calib data file")
            
     #%% TvP all plot      
    if eval_TvP:
        tools.plot_TvP(hist_dic, c,  rmse_table, N_test=N_test,
                       discretized_parameters=discretized_parameters, plotting_map=False, title=network_name,
                       figname=figurename_dic['TvP'], return_figure=False)
        
        if VERBOSE:
            print("Saved TvP figure")
        
        #%% TvP MAP plot
        tools.plot_TvP(map_hist_dic, c,  rmse_table, N_test=N_test,
                       discretized_parameters=discretized_parameters, plotting_map=True, title=network_name,
                       figname=figurename_dic['TvP_MAP'], return_figure=False)
        
        if VERBOSE:
            print("Saved TvP MAP figure")
        
    
    #%% plot calibration
    if eval_calib:
        tools.plot_calibration(calib_table, title=network_name, figname=figurename_dic['calib'], return_figure=False)
        
        if VERBOSE:
            print("Saved calibration plot")
     
    plt.close()
#%% Main  
        
if __name__=='__main__':
    
    # Parse optional command line arguments
    parser = ArgumentParser()
    # parser.add_argument("-c", "--config", dest="config_file", default='./config.py',
    #                     help="Run with specified config file as basis.")
    
    parser.add_argument('config_file', help="Run with specified config file as basis.")
    parser.add_argument('-d','--device', required=False, default=None, help="device for network")
    parser.add_argument('-l','--lsig', required=False, default=None, help="Const log (sigma) for Noise-Net evaluation. If this is set, the filenames change.")
    parser.add_argument('-g', '--group_size', type=int, default=GROUP_SIZE, help="# of obs per one posterior processing (Default=400)")
    parser.add_argument('-L','--log', required=False, default=True, help="Save logfile T/F")
    
    args = parser.parse_args()
    
    if args.log =='True' or  args.log=='1':
        savelog=True
    elif args.log is not True:
        savelog = False
    else:
        savelog = True
    
    GROUP_SIZE = args.group_size
        
    # if savelog:
    #     logfile = os.path.basename(args.config_file).replace('.py','_evaluation.log').replace('c_','')
    #     sys.stdout = Logger(logfile, log_mode = LOG_MODE)
    
    # Import default config
    config_file = args.config_file
        
    c = read_config_from_file(config_file)
    
  
        
    # for Noise-Net, if specific lsig is set
    if c.prenoise_training==True and args.lsig is not None:
        lsig_set = float(args.lsig)
        print("Use specific lsig (%f) for this evaluation"%lsig_set)
    else:
        lsig_set = None
        
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
        
        logfile = logpath.replace('/','.') + os.path.basename(c.filename)+'_eval.log'
        
        # sys.stdout = Logger(logfile, log_mode=LOG_MODE, mode="a") # continue logging
        logger = Logger(logfile, log_mode=LOG_MODE, mode="a")
        
    astro = DataLoader(c)
    
    if args.device is not None:
        device = args.device
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
    
    # _ = torch.Tensor([0]).to(c.device)
    astro.device = c.device
   
    print("==================== CINN NETWORK SETTING =================")
    print("cINN_config:", config_file)
    print("# of parameters:", c.x_dim)
    print("# of observables:", c.y_dim_in)
    print("Database:", c.tablename)
    print("using device:", c.device)
    print("===========================================================")
    
    
    evaluate(c, astro=astro, lsig_fix=lsig_set)
    print("Finished evaluation: %s"%config_file)
    
    # move log file to path
    if savelog:
        # sys.stdout.close()  # close log file
        # sys.stdout = sys.__stdout__ # stop logging
        logger.close()
        
        new_logfile = logpath + os.path.basename(c.filename) +'_eval.log'
        os.system("mv {} {}".format(logfile, new_logfile)  )

    

        