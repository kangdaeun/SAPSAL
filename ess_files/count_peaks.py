#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 16:46:44 2023

@author: daeun

Count peaks from fitting result of posterior distribution

- Read fitparam and posterior distribution

"""

import numpy as np
import os, sys, glob
from time import time
from astropy.io import ascii
from astropy.table import Table

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from argparse import ArgumentParser
import gc

#%%
###########
## Setup ##
###########

# hyper params
use_l2_norm=True # nspeak separataion condition
plot_figure = True # plotting
plot_N_cluster = True
save_figure = True

hatch_list = ['/', '\\', '-', '+', '.', '*']
xlabelsize = "large"
ylabelsize = "medium"
xticksize = 'medium'
yticksize = 'small'
legendsize = 'small'

title_dic = {
    'logTeff': "log T$_{\mathrm{eff}}$ [K]",
    'Teff': 'T$_{\mathrm{eff}}$ [K]',
    'logG': 'log g [cm s$^{-2}$]',
    'A_V': 'A$_{\mathrm{V}}$ [mag]',
    'library': 'Library',
    'veil_r': 'Veiling factor',
}


#%%

def gauss_one(x, amp, center, sigma):
    return amp*np.exp(-(x-center)**2/(2*sigma**2))

def gauss_one_offset(x, offset, amp, center, sigma):
    return offset+amp*np.exp(-(x-center)**2/(2*sigma**2))
    
def gauss_first_derivative(x, amp, center, sigma):  
    return (-amp*(x-center)/(sigma**2.))*np.exp( -(x-center)**2. / (2*sigma**2)   )
 
def gauss_second_derivative(x, amp, center, sigma):
    return amp/sigma**2 * ( (x-center)**2./sigma**2 - 1 )*np.exp( -(x-center)**2./(2*sigma**2) )

def multi_gauss_derivative_value(*params, first=True, second=True, zero=True):
    
    fit_params = list(params)
    x = np.array(fit_params.pop(0))

    if len(np.shape(fit_params)) == 2:
        fit_params = fit_params[0]
    
    p0 = fit_params[0]
    
    total0 = np.zeros(len(x))+float(p0)
    total1 = np.zeros(len(x))
    total2 = np.zeros(len(x))

    for group in range(len(fit_params[1:])//3):
        p1,p2,p3 = fit_params[1+group*3:1+group*3+3]
        total0 += gauss_one(x, p1, p2, p3)
        total1 += gauss_first_derivative(x, p1, p2, p3)
        total2 += gauss_second_derivative(x, p1, p2, p3)
    
    result = {}
    if zero == True: result[0] = total0
    if first == True: result[1] = total1
    if second == True: result[2] = total2

    return result

def multi_gauss(x, fit_params):
    total0 = np.zeros(len(x)) + float(fit_params[0])
    for group in range(len(fit_params[1:])//3):
        p1,p2,p3 = fit_params[1+group*3:1+group*3+3]
        total0 += gauss_one(x, p1, p2, p3)
    return total0

def multi_gauss_first_derivative(x, fit_params):
    total1 = np.zeros(len(x))
    for group in range(len(fit_params[1:])//3):
        p1,p2,p3 = fit_params[1+group*3:1+group*3+3]
        total1 += gauss_first_derivative(x, p1, p2, p3)
    return total1
    
def multi_gauss_second_derivative(x, fit_params):
    total2 = np.zeros(len(x))
    for group in range(len(fit_params[1:])//3):
        p1,p2,p3 = fit_params[1+group*3:1+group*3+3]
        total2 += gauss_second_derivative(x, p1, p2, p3)
    return total2

def l1_norm(sig1, sig2):
    return abs(sig1+sig2)
def l2_norm(sig1, sig2):
    return np.sqrt(sig1**2 + sig2**2)

def separate_condition(c1, c2, sig1, sig2, use_l1_norm=False, use_l2_norm=True):
    if use_l1_norm: f = l1_norm
    if use_l2_norm: f = l2_norm
    
    if abs(c1-c2) > f(sig1, sig2):
        return 1
    else: 
        return 0


def count_peaks(fit_params, raw_data, use_l2_norm=True):

    n_mode = 0; n_vpeaks = 0; n_sep = 0;
    xpeaks = np.zeros(1)+np.nan; ypeaks = np.zeros(1)+np.nan;

    # n_mode : number of Gaussian components 
    N_component = len(fit_params[1:])//3
    n_mode = N_component # done for nmode_table

    if N_component<=1:
        n_vpeaks=n_mode
        n_speaks=n_mode
        
        if N_component==1:
            xpeaks = np.array([fit_params[2]])
            ypeaks = multi_gauss(xpeaks, fit_params)
    else:
        # N > 1
        for i in range(N_component):
            if i==0:
                xfit = np.linspace(fit_params[3*i+2]-5*fit_params[3*i+3], fit_params[3*i+2]+5*fit_params[3*i+3], 100)
            else:
                xfit = np.append(xfit, np.linspace(fit_params[3*i+2]-5*fit_params[3*i+3], fit_params[3*i+2]+5*fit_params[3*i+3], 100) )

        xmin = np.nanmin(raw_data); xmax = np.nanmax(raw_data)
        if xmin < xfit[0]: xfit=np.append(np.array([xmin]), xfit)
        if xmax > xfit[1]: xfit=np.append(xfit, np.array([xmax]))
        xfit = np.sort(xfit)

        # Visible peak (1st derivati)
        dic = multi_gauss_derivative_value(xfit, fit_params, zero=True, first=True, second=True)
        yfit, y1, y2 = dic[0], dic[1], dic[2]
        roi = (y1[1:]*y1[:-1])<0
        roi2 = (y2[1:]<0) * (y2[:-1] < 0)
        a=np.where(roi*roi2)[0]
        xpeaks = 0.5*(xfit[a]+xfit[a+1])
        ypeaks = multi_gauss(xpeaks, fit_params)
        n_vpeaks = len(xpeaks)

        # Separate peak (gauss center, sigma condition)
         # sort by center
        fit_params_2d = fit_params[1:].reshape(-1, 3)
        fit_params_2d = fit_params_2d[ np.argsort(fit_params_2d[:,1])  ]
        fit_params = np.append(fit_params[0], fit_params_2d.reshape(-1))

        sep_bool = np.zeros(shape=(N_component, N_component))
        for i in range(N_component-1):
            for j in range(i+1, N_component):
                sep_bool[i, j] = separate_condition(*fit_params[[3*i+2, 3*j+2, 3*i+3, 3*j+3]], use_l2_norm=use_l2_norm, use_l1_norm=np.invert(use_l2_norm))
                sep_bool[j, i] = sep_bool[i,j]

        n_sep = 1
        while sep_bool.shape[0]>1:
            n = sep_bool.shape[1]

            s = np.sum(sep_bool, axis=0)
            # separated to all the others
            a = np.where(s == n-1)[0] # separated to all
            b = np.where(s == 0)[0] # blended to all
            if len(a)>0:
                n_sep += 1
                sep_bool = np.delete(sep_bool, a[0], 0) # delete row
                sep_bool = np.delete(sep_bool, a[0], 1) # delete column

            elif len(b)>0:
                sep_bool = np.delete(sep_bool, b[0], 0) # delete row
                sep_bool = np.delete(sep_bool, b[0], 1) # delete column
            else:
                c = np.where(s == np.max(s))[0]
                n_sep += 1 
                sep_bool = np.delete(sep_bool, c[0], 0) # delete row
                sep_bool = np.delete(sep_bool, c[0], 1) # delete column

        n_speaks = n_sep

    result = {'n_mode':n_mode, 'n_vpeaks':n_vpeaks, 'n_speaks': n_speaks, 'vpeaks_x': xpeaks, 'vpeaks_y': ypeaks}

    return result

#%%
##########
## MAIN ##
##########

if __name__=='__main__':
    
    # Parse optional command line arguments
    parser = ArgumentParser()
    # parser.add_argument("-c", "--config", dest="config_file", default='./config.py',
    #                     help="Run with specified config file as basis.")
    
    
    parser.add_argument('-pd', '--post_dir', required=True, default=None, help="Dirpath to posterior files")
    parser.add_argument('-fd', '--fit_dir', required=True, default=None, help="Dirpath to fitparam files")
    
    parser.add_argument('-sd', '--save_dir', required=True, help="directory to save fit_params and plots" )
    parser.add_argument('-rn', '--renew', required=False, default=False, help="Renewal of existing results")
    
    
    # Import default config
    args = parser.parse_args()
    
    if args.renew == "True" or args.renew=="1":
        renewal = True
        print("Renew and overwrite the existing results")
    else:
        renewal = False
        
    # save plots and count information here
    save_dir = args.save_dir
    if save_dir[-1]!='/':
        save_dir += '/'
    if not os.path.exists(save_dir):
        os.system('mkdir -p {}'.format(save_dir))
    if save_figure:
        os.system('mkdir -p {}'.format(save_dir+'figs/'))
    print("Save data here:",save_dir)
        
    # ================ Read fitparam filenames ====================
    file_list = []
    if args.fit_dir is not None: # run multiple posterior files
        # find all post files in the path
        file_list = sorted(glob.glob(args.fit_dir+'fitparam_*.txt'))
        if len(file_list)==0:
            sys.exit("No fitparam files in fit_dir (%s)"%args.fit_dir)
        else:
            file_list = sorted(file_list)
            print("%d fitparam files detected"%(len(file_list)))
    else:
        sys.exit("No fit_dir specified")
        
    #============== Check existence of posterior directory ============
    if not os.path.exists(args.post_dir):
        sys.exit("Posterior directory (%s) does not exist"%args.post_dir)
    post_dir = args.post_dir
        
    # Run
    t_start = time()
    print("Count peaks for %d models"%len(file_list))
    
    nmode_table = []
    nvpeak_table = []
    nspeak_table = []
    
    for i_model, file in enumerate(file_list):
        
        print('Start model: ',i_model)
        
        file_code = os.path.basename(file).replace("fitparam_","").replace(".txt","")
        data = ascii.read(file, format='commented_header', delimiter='\t')
        post_file = post_dir+'post_'+file_code+'.dat'
        if os.path.exists(post_file):
            post = ascii.read(post_file, format='commented_header', delimiter='\t')  
        else:
            sys.exit("Cannot find posterior file (%s)"%post_file)
        
        params_to_fit = data.colnames
        parameter_names = post.colnames
        
        if np.sum(np.in1d(parameter_names, params_to_fit))!=len(params_to_fit):
            sys.exit("posterior does not contain fit parameter")
        
    
        n_mode = np.zeros(len(params_to_fit)+2).astype(int)
        n_vpeaks = np.zeros(len(params_to_fit)+2).astype(int)
        n_speaks = np.zeros(len(params_to_fit)+2).astype(int)
        
        # additional information (at the end) in the mode table (field, n)
        n_mode[-2] = int(file_code.split('_')[1])
        n_mode[-1] = int(file_code.split('_')[2])
        
        if plot_figure:
            nrow = np.sqrt(len(params_to_fit)).astype(int)
            if len(params_to_fit)%nrow ==0:
                ncol = len(params_to_fit)//nrow
            else:
                ncol = len(params_to_fit)//nrow + 1
    
            fig, axis = plt.subplots(nrow, ncol,figsize=[3.5*ncol, 3.2*nrow], tight_layout=True)
            axis=axis.ravel()
        
        for i_param, param in enumerate(params_to_fit):
            
            fit_params = data[param]
            fit_params = np.array(fit_params[np.isfinite(fit_params)])
    
            raw = post[param]
            
            
            if len(fit_params)==0:
                fitting_success = False
            else:
                fitting_success = True
               
            # Count fit (only if fitting succeded)
            if fitting_success:
                result = count_peaks(fit_params, raw, use_l2_norm=True)
                # {'n_mode':n_mode, 'n_vpeaks':n_vpeaks, 'n_speaks': n_speaks, 'vpeaks_x': xpeaks, 'vpeaks_y': ypeaks}
                n_mode[i_param] = result['n_mode']
                n_vpeaks[i_param] = result['n_vpeaks']
                n_speaks[i_param] = result['n_speaks']

                # location of vpeaks
                vpeaks_x = result['vpeaks_x']
                vpeaks_y = result['vpeaks_y']
            else:
                n_mode[i_param] = 0
                n_vpeaks[i_param] = 0
                n_speaks[i_param] = 0
            
            if plot_figure:
                ax = axis[i_param]
                yhis, _xhis = np.histogram(raw, bins=np.round(np.sqrt(len(raw))).astype(int), density=True)
                xhis = 0.5*(_xhis[1:]+_xhis[:-1])
                ax.step(xhis, yhis, where='mid', lw=1.5, label='raw')
                
                if fitting_success:
                    xfit = np.linspace(np.nanmin(raw), np.nanmax(raw), 100)
                    yfit = multi_gauss(xfit, fit_params)
                    ax.plot(xfit, yfit, lw=1.5, label='fit')
                
                if n_mode[i_param] > 0:
                    ax.text(0.05, 0.9, '(%d, %d, %d)'%(n_mode[i_param], n_vpeaks[i_param], n_speaks[i_param]), transform=ax.transAxes, ha='left',fontsize=legendsize,
                           bbox=dict(boxstyle='round', facecolor='w', alpha=0.6, edgecolor='silver'))
                    # visible peak
                    ax.plot(vpeaks_x, vpeaks_y, 'o', color='red', ms=6, mec='k') 
                    
                    # Gaussian components
                if n_mode[i_param] > 1:
                    mode_handles = []
                    # print(n_mode[i_param])
                    for i in range(n_mode[i_param]):
                        yg = gauss_one(xfit, *fit_params[3*i+1: 3*(i+1)+1])
                        l=ax.plot(xfit, yg, ls='--', lw=1,label='mode %d'%(i+1))
                        ax.fill_between(xfit, yg, alpha=0.1, hatch=hatch_list[i], color=l[0].get_color(),)
    
                if i_param==0:
                    ax.legend(loc='upper right', fontsize=legendsize)
                    
                ax.tick_params(axis='y',which='both',labelsize='medium')
            
                ax.set_xlabel(title_dic[param], size=xlabelsize)
                ax.set_ylabel('p(x)', size=ylabelsize)
                ax.yaxis.set_major_locator(ticker.MaxNLocator(2))
                ax.tick_params(axis='y', labelsize=yticksize)
                ax.tick_params(axis='x', labelsize=xticksize)
                ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
                
#         if plot_figure:
#             if i_param+1 < len(axis):
#                 for jj in range(i_param+1, len(axis)):
#                     axis[jj].axis("off")
                
                
        nmode_table.append(n_mode)
        nvpeak_table.append(n_vpeaks)
        nspeak_table.append(n_speaks)
             
                
        if plot_figure:
            if i_param+1 < len(axis):
                for jj in range(i_param+1, len(axis)):
                    axis[jj].axis("off")
            
            if save_figure:
                fig_file = save_dir + 'figs/Countfig_'+file_code+'.png'
                fig.savefig(fig_file, dpi=250)
            plt.close()
            
        
                
    nmode_table = Table(np.array(nmode_table), names=parameter_names+['field','n'])
    nvpeak_table = Table(np.array(nvpeak_table), names=parameter_names+['field','n'])
    nspeak_table = Table(np.array(nspeak_table), names=parameter_names+['field','n'])
    for param in [k for k in nmode_table.colnames if k not in params_to_fit]:
        nvpeak_table[param] = nmode_table[param]
        nspeak_table[param] = nmode_table[param]
        
    ascii.write(nmode_table, save_dir + 'nmode_table.txt', format='commented_header', delimiter='\t', overwrite=True)
    ascii.write(nvpeak_table, save_dir + 'nvpeak_table.txt', format='commented_header', delimiter='\t', overwrite=True)
    ascii.write(nspeak_table, save_dir + 'nspeak_table.txt', format='commented_header', delimiter='\t', overwrite=True)
    
    t_end = time()
    print('Finished (%.2f min)'%( (t_end-t_start)/60. ))   
    
    gc.collect()