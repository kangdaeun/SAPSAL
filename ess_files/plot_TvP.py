#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 12:03:30 2022

@author: daeun

Validation code
- True vs Post
- True vs MAP (median) -> not good for degenerate predictions

Save histogram info as pckl
Save figures
(all saved in output/)

"""

import numpy as np
import matplotlib.pyplot as plt

import os, sys
import matplotlib.cm as cm
from pathlib import Path
from time import time
import pickle


from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patheffects as PathEffects

import matplotlib.colors as clr


import matplotlib.ticker as ticker


from cINN_set.cINN_config import *
from cINN_set.data_loader import *

import copy


config_file = sys.argv[1]
c = read_config_from_file(config_file)
# print(config_file)

filename = c.filename + '_TvP.pdf'

if len(sys.argv)==3:
    device = sys.argv[2]
    # print('Change device from %s to %s'%(c.device, device))
    c.device = device

print("==================== CINN NETWORK SETTING =================")
print("cINN_config:", config_file)
print("# of parameters:", c.x_dim)
print("# of observables:", c.y_dim_in)
print("Database:", c.tablename)
print("using device:", c.device)
print("===========================================================")

astro = DataLoader(c)
test_set, train_set = astro.get_splitted_set(rawval=True, smoothing = False, smoothing_sigma = None)
param_test = test_set[0]; obs_test = test_set[1]

#%%
hist_range_dic = {}

for i_param, param in enumerate(astro.x_names):
    
    xrange = [np.min(test_set[0][:,i_param]), np.max(test_set[0][:,i_param])]
    yrange = xrange
    
    hist_range_dic[param] = (xrange, yrange)

#%%
"""
Caculate all post histogram and MAP values once 
Save both histograms
"""

# Setting for p(x) sampling
group_size = 200 # n(obs) per one iteration
n_group = np.ceil(len(param_test)/group_size).astype(int)
N_pred = 4096 # latent variable sampling

# Setting for histogram
nbin=100 # histogram bins for True vs All
if len(obs_test) < 1e4:
    nbin = np.ceil(np.sqrt(len(obs_test))).astype(int)

# special parameters
discretized_parameters = []

# save setting
hist_name = filename.replace('.pdf','.pkl')
# Read hist_dic from pickle: hist_dic, map_hist_dic
if os.path.exists(hist_name):
    print('Histogram file exists, Read histogram')
    with open(hist_name, 'rb') as hf:
        dic = pickle.load(hf)
        hist_dic = dic['hist_dic']
        map_hist_dic = dic['map_hist_dic']

else:
    print('No histogram file exists. Get posteriors')
    t1=time()
    hist_dic = {} # save histogram info of True vs All
    for i_group in range(n_group):
    # for i_group in range(10):



        obs_group = obs_test[i_group*group_size:(i_group+1)*group_size]
        param_group = param_test[i_group*group_size:(i_group+1)*group_size]

        post_list = astro.exp.get_posterior(obs_group, c, N=N_pred, use_group=True, group=group_size,
                                      return_llike=False, quiet=True)

        # calculate MAP (median) 
        med = np.nanmedian(post_list, axis=1)
        if i_group==0:
            med_post = med
        else:
            med_post = np.vstack([med_post, med])

        # calculate histogram (True vs All) for this group
        for i_param, param in enumerate(c.x_names):
            true1d = param_group[:,i_param].reshape(-1,1).repeat(N_pred,axis=1).ravel()
            post1d = post_list.reshape(-1, len(c.x_names))[:,i_param]

            bins = nbin
            # add condition for discretized parameters
            # if param=='N_cluster':
            #     bins=[np.arange(0.5, np.max(test_table[param].data)+0.51, 1.), np.arange(0.5, np.max(test_table[param].data)+0.51, 1.)]
            # elif param=='phase':
            #     bins=[np.arange(-0.5, 3.51, 1.), np.arange(-0.5, 3.51, 1.)]
            # else:
            #     bins = nbin


            if i_group==0:
                xrange, yrange = hist_range_dic[param]
                H, xedges, yedges = np.histogram2d(true1d, post1d, bins=bins, range=(xrange,yrange))
                hist_dic[param] = {'H': H,'xedges':xedges, 'yedges': yedges  }

            else:
                H, _1, _2 = np.histogram2d(true1d, post1d, bins=(hist_dic[param]['xedges'], hist_dic[param]['yedges']) )
                hist_dic[param]['H'] += H

        if (i_group%10==0):
            print('{:d}th group: {:.2f} min'.format(i_group, (time()-t1)/60. ))
    
    # MAP histogram (after for loop)
    map_hist_dic = {}
    for i_param, param in enumerate(c.x_names):
        xtrue = param_test[:,i_param]
        map_values = med_post[:,i_param]
        xrange, yrange = hist_range_dic[param]

        bins = nbin
        # add condition for discretized parameters
        # if param=='N_cluster':
        #     bins=[np.arange(0.5, np.max(test_table[param].data)+0.51, 1.), np.arange(0.5, np.max(test_table[param].data)+0.51, 1.)]
        # elif param=='phase':
        #     bins=[np.arange(-0.5, 3.51, 1.), np.arange(-0.5, 3.51, 1.)]
        # else:
        #     bins = nbin

        H, xedges, yedges = np.histogram2d(xtrue, map_values,
                                          bins=bins, range=(xrange,yrange))
        map_hist_dic[param] = {'H': H,'xedges':xedges, 'yedges': yedges  }
    print('Time taken: {:.2f} min'.format((time()-t1)/60. ))
    
    """
    File to save
    - histogram info of True Vs All
    - histogram info of True vs MAP
    """
    # Save histogram 
    s_f = open(hist_name, 'wb')
    pickle.dump({'hist_dic':hist_dic, 'map_hist_dic':map_hist_dic}, s_f)
    s_f.close()
    print('Saved histogram file')

#%%   
# currently not used
infos = {
                'config_name': 'config: %s'%c.config_file,
                'model_code': c.model_code,
                'adam_betas': 'Adb: (%.2g,%.2g)'%(c.adam_betas),
                'batch_size': 'B: %d'%int(c.batch_size),
                'n_blocks': '$N_{\mathrm{block}}$: %d'%int(c.n_blocks),
                'gamma': '$\gamma_{\mathrm{decay}}$: %.3g'%c.gamma,
                'lr_init': '$Lr_{\mathrm{init}}$: %.3g'%c.lr_init,
                'l2_weight_reg': '$L2_{\mathrm{reg}}$: %.1e'%c.l2_weight_reg,
                'meta_epoch': '$Sc_{\mathrm{epoch}}$: %d'%int(c.meta_epoch), 
                'test_frac': '$f_{\mathrm{test}}$: %.2g'%(c.test_frac),
                'n_epochs': '$N_{\mathrm{epoch}}$: %d'%(c.n_epochs),
                            }
#%% 

"""
Calculate confusion matrix for discretized parameters (All)
"""
conf_matrix = {}

for param in discretized_parameters:
    if param in astro.x_names:
        H, xedges, yedges = hist_dic[param]['H'].copy(), hist_dic[param]['xedges'], hist_dic[param]['yedges']
        H = H.transpose()
        conf = H.copy()*0.0
    
        for i in range(H.shape[1]):
            conf[:,i] = H[:,i]/np.sum(H[:,i])*100
        
        conf_matrix[param] = conf
        
"""
All posterior estimates plot
"""   

H_min = 10 # 보통
if len(obs_test) < 1e4:
    H_min = 5
# set figure size and grid

# nrow는 4의 배수로 끊김 ~4:1, ~8:2, ~12:3
nrow = np.ceil(len(c.x_names)/4).astype(int)
ncol = np.ceil(len(c.x_names)/nrow).astype(int)
figsize = [3.1*ncol, 4*nrow]
fig, axis = plt.subplots(nrow, ncol, figsize=figsize)
axis = axis.ravel()



for i_param, param in enumerate(c.x_names):
    ax = axis[i_param]
    
    H, xedges, yedges = hist_dic[param]['H'].copy(), hist_dic[param]['xedges'], hist_dic[param]['yedges']
    
    roi = H < H_min
    Hsum = np.nansum(H)
    H[roi] = np.nan
    H = H/Hsum
    
    cmap = copy.copy(cm.Oranges)
    # cmap = copy.copy(cm.gnuplot)
    cmap.set_bad("grey", 0.2)
    
    ims=ax.imshow( H.transpose(), origin='lower', extent = (xedges[0], xedges[-1], yedges[0], yedges[-1]), 
                  aspect='auto',
             cmap=cmap)#, norm=clr.LogNorm())
    
    axins = ax.inset_axes( [0.07, 0.89, 0.5, 0.03 ])
    cbar=fig.colorbar(ims, cax=axins, orientation="horizontal")
    cbar.minorticks_off()
    axins.tick_params(axis='both',which='both',labelsize='medium', direction='out', length=2)
    axins.set_title('Fraction', size='medium')
    # axins.xaxis.set_major_locator(ticker.LogLocator(numticks=3))
    axins.xaxis.set_major_locator(ticker.MaxNLocator(3))
    
    if param in conf_matrix.keys():
        conf = conf_matrix[param]
        
        xp = 0.5*(xedges[1:]+xedges[:-1])
        yp = 0.5*(yedges[1:]+yedges[:-1])
        
        for ix, x in enumerate(xp):
            for iy, y in enumerate(yp):
                if np.isfinite(H[ix,iy]):
                    string = '%.2g'%(conf[iy,ix])
                    if (conf[iy,ix] < 1): string = '%.1f'%(conf[iy,ix])
                    if conf[iy,ix] >= 99.5: string = '100'
                    ax.text(x, y, string,  color='k', fontsize=8.5, ha='center', va='center', path_effects=[PathEffects.withStroke(linewidth=2, foreground='w')] )
    
    # 1:1 
    xr = ax.get_xlim()
    yr = ax.get_ylim()
    mini = np.min(xr+yr); maxi = np.max(xr+yr)
    xx=np.linspace(mini, maxi, 500)
    ax.plot(xx,xx,'-',color='navy', lw=0.7)
    ax.set_xlim(xr); ax.set_ylim(yr)
    
    
    ax.set(title=param, xlabel=r'$X^{\mathrm{True}}$', ylabel=r"$X^{\mathrm{Post}}$")
    
    
fig.tight_layout()
fig.savefig(filename)
#%%
"""
Calculate confusion matrix for discretized parameters (MAP)
"""
conf_matrix = {}

for param in discretized_parameters:
    if param in astro.x_names:
        H, xedges, yedges = map_hist_dic[param]['H'].copy(), map_hist_dic[param]['xedges'], map_hist_dic[param]['yedges']
        H = H.transpose()
        conf = H.copy()*0.0
    
        for i in range(H.shape[1]):
            conf[:,i] = H[:,i]/np.sum(H[:,i])*100
        
        conf_matrix[param] = conf

"""
MAP posterior estimates plot
"""
H_min = 5 # 보통
if len(obs_test) < 1e4:
    H_min = 1
# set figure size and grid

# nrow는 4의 배수로 끊김 ~4:1, ~8:2, ~12:3
nrow = np.ceil(len(c.x_names)/4).astype(int)
ncol = np.ceil(len(c.x_names)/nrow).astype(int)
figsize = [3.1*ncol, 4*nrow]
fig, axis = plt.subplots(nrow, ncol, figsize=figsize)
axis = axis.ravel()


for i_param, param in enumerate(c.x_names):
    ax = axis[i_param]
    
    H, xedges, yedges = map_hist_dic[param]['H'].copy(), map_hist_dic[param]['xedges'], map_hist_dic[param]['yedges']
    
    roi = H < H_min
    Hsum = np.nansum(H)
    H[roi] = np.nan
    H = H/Hsum
    
    cmap = copy.copy(cm.Oranges)
    # cmap = copy.copy(cm.gnuplot)
    cmap.set_bad("grey", 0.2)
    
    ims=ax.imshow( H.transpose(), origin='lower', extent = (xedges[0], xedges[-1], yedges[0], yedges[-1]), 
                  aspect='auto',
             cmap=cmap)#, norm=clr.LogNorm())
    
    axins = ax.inset_axes( [0.07, 0.89, 0.5, 0.03 ])
    cbar=fig.colorbar(ims, cax=axins, orientation="horizontal")
    cbar.minorticks_off()
    axins.tick_params(axis='both',which='both',labelsize='medium', direction='out', length=2)
    axins.set_title('Fraction', size='medium')
    # axins.xaxis.set_major_locator(ticker.LogLocator(numticks=3))
    axins.xaxis.set_major_locator(ticker.MaxNLocator(3))
    
    if param in conf_matrix.keys():
        conf = conf_matrix[param]
        
        xp = 0.5*(xedges[1:]+xedges[:-1])
        yp = 0.5*(yedges[1:]+yedges[:-1])
        
        for ix, x in enumerate(xp):
            for iy, y in enumerate(yp):
                if np.isfinite(H[ix,iy]):
                    string = '%.2g'%(conf[iy,ix])
                    if (conf[iy,ix] < 1): string = '%.1f'%(conf[iy,ix])
                    if conf[iy,ix] >= 99.5: string = '100'
                    ax.text(x, y, string,  color='k', fontsize=8.5, ha='center', va='center', path_effects=[PathEffects.withStroke(linewidth=2, foreground='w')] )
    
    # 1:1 
    xr = ax.get_xlim()
    yr = ax.get_ylim()
    mini = np.min(xr+yr); maxi = np.max(xr+yr)
    xx=np.linspace(mini, maxi, 500)
    ax.plot(xx,xx,'-',color='navy', lw=0.7)
    ax.set_xlim(xr); ax.set_ylim(yr)
    
    
    ax.set(title=param, xlabel=r'$X^{\mathrm{True}}$', ylabel=r"$X^{\mathrm{MAP}}$")
    
    
fig.tight_layout()
fig.savefig(filename.replace('.pdf','_MAP.pdf'))
