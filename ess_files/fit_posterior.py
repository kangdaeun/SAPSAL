#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 13:57:31 2023

@author: daeun

fit posterior : CPU

1) read posterior files (do not run posterior here)
    - posterior filename format should be: post_XXXXXX.dat
2) fit 1D posterior distributions with Gaussians max. 6
3) save plot and fitting params

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
from scipy import optimize

#%%
###########
## Setup ##
###########

params_to_fit = None # if None: fit all parameters in the file
params_to_plot = None #  if None: plot all parameters in the file

plot_figure = True
save_figure = True



# Figure setup
hatch_list = ['/', '\\', '-', '+', '.', '*']
xlabelsize = "x-large"
ylabelsize = "large"
xticksize = 'large'
yticksize = 'medium'


title_dic = {
    'logTeff': "log T$_{\mathrm{eff}}$ [K]",
    'Teff': 'T$_{\mathrm{eff}}$ [K]',
    'logG': 'log g [cm s$^{-2}$]',
    'A_V': 'A$_{\mathrm{V}}$ [mag]',
    'library': 'Library',
    'veil_r': 'Veiling factor',
}


#%%
def multi_gauss(*params):
    """
    Supported ways to call multi_gauss
    multi_gauss(x, fit_params)
    multi_gauss((x, fit_parm1, fit_parm2, fit_parm3...fit_parmN))
    This makes it possible to use as in a fitting routine and plotting.
    """

    fit_params = list(params)

    x = np.array(fit_params.pop(0))

    if len(np.shape(fit_params)) == 2:
        fit_params = fit_params[0]
    
    p0 = fit_params[0]
    total = np.zeros(len(x))
    total += float(p0)

    for group in range(len(fit_params[1:])//3):
        p1,p2,p3 = fit_params[1+group*3:1+group*3+3]
        total += p1*np.exp(-(x-p2)**2/(2*p3**2))

    return total


def pguess_multigauss(x, y, amp_guess=None, center_guess=None, sigma_guess=None):

        if (amp_guess == None) or (center_guess == None):
            i_max = np.argmax(y)

        if (center_guess == None):
            # # Get its value
            center_guess = x[i_max]

        if (amp_guess == None):
            # # Get it's amplitude
            amp_guess = y[i_max]

        if sigma_guess == None:
            sigma_guess = 1.0
        
        return [amp_guess, center_guess, sigma_guess]
    
    
def bound_multigauss(amp_up=np.inf, amp_low=-np.inf, center_up=np.inf, center_low=-np.inf, sigma_up=np.inf, sigma_low=-np.inf):

    
        low = [amp_low, center_low, sigma_low]
        up = [amp_up, center_up, sigma_up]
        
        
        return low, up
    
    
def calculate_fd_nbin(data):
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    h = 2*iqr*(len(data))**(-1./3)
    bins = np.round( (np.nanmax(data)-np.nanmin(data))/h  ).astype(int)
    return bins


# Fitting
def fit_hist(yhis, xhis, N_max=6, initial_offset = 0.0, fresi_success = 0.03, fresi_add=0.05):
    """
    Fitting binned posterior distributions = histrogram

    Parameters
    ----------
    yhis : TYPE
        histrogram yvalue
    xhis : TYPE
        dim(xhis) = dim(yhis)
    N_max : TYPE, optional
        The maximum number of Gaussian components in one distribution. The default is 6.
    initial_offset : TYPE, optional
        Initial fitting value for offset. The default is 0.0.
    fresi_success : TYPE, optional
        fraction of residual_sum over hist_sum. condition for finishing fitting. The default is 0.03.
    fresi_add : TYPE, optional
        condition for adding more Gaussian component. The default is 0.05.

    Returns
    -------
    popt : TYPE
        fitting parameters.
    pcov : TYPE
        DESCRIPTION.

    """
    # We calculate the integral of the histogram. Not just the number of points, but weighted by the ... weight.
    hist_sum = np.sum(yhis)
    # hist_sum = np.trapz(yhis, xhis)
    residual = yhis
    residual_rms = np.sqrt(np.sum(np.square(residual)))
    # residual_rms = np.sqrt(np.trapz(abs(residual), xhis))

    # N_max = 6
    
    # Initialize pguess with the estimate for the continuum level (offset from 0)
    pguess = [initial_offset]
    b_low = [-np.inf]
    b_up = [np.max(yhis)]
    # b_low = [-1e-5]
    # b_up = [1e-5]

    N_components = 0
    sigma0 = xhis[1] - xhis[0]

    while (residual_rms / hist_sum > fresi_success) and (N_components < N_max):
        N_components += 1            
        # Use pguess_multiguass to add a componenent
        pguess += pguess_multigauss(xhis, residual, sigma_guess=sigma0)

        b = bound_multigauss(amp_low = 0, amp_up = 1.5*np.max(yhis), 
                             center_low = np.min(xhis), center_up=np.max(xhis),
                             sigma_low = 0, sigma_up = xhis[-1]-xhis[0])   
        b_low += b[0]
        b_up += b[1]

        popt, pcov = optimize.curve_fit(multi_gauss, xhis, yhis, p0=np.array(pguess), bounds= ( b_low, b_up ) )

        residual = yhis - multi_gauss(xhis, popt)
        residual_rms = np.sqrt(np.sum(np.square(residual)))
        # residual_rms = np.sqrt(np.trapz(abs(residual), xhis))

        if residual_rms / hist_sum > fresi_add:
            pguess = list(popt)
            
    return (popt, pcov)

#%%
##########
## MAIN ##
##########

if __name__=='__main__':
    
    # Parse optional command line arguments
    parser = ArgumentParser()
    # parser.add_argument("-c", "--config", dest="config_file", default='./config.py',
    #                     help="Run with specified config file as basis.")
    
    
    parser.add_argument('-f', '--file', required=False, default=None, help="posterior file")
    parser.add_argument('-fd', '--file_dir', required=False, default=None, help="Dirpath to posterior files")
    
    parser.add_argument('-sd', '--save_dir', required=True, help="directory to save fit_params and plots" )
    parser.add_argument('-rn', '--renew', required=False, default=False, help="Renewal of existing results")
    
    
    # Import default config
    args = parser.parse_args()
    
    if args.renew == "True" or args.renew=="1":
        renewal = True
        print("Renew and overwrite the existing results")
    else:
        renewal = False
        
    # save plots and fit_params here
    save_dir = args.save_dir
    if save_dir[-1]!='/':
        save_dir += '/'
    if not os.path.exists(save_dir):
        os.system('mkdir -p {}'.format(save_dir))
    if save_figure:
        os.system('mkdir -p {}'.format(save_dir+'figs/'))
    print("Save data here:",save_dir)
        
    # ================ Read posterior filenames ====================
    pfile_list = []
    if args.file_dir is not None: # run multiple posterior files
        # find all post files in the path
        pfile_list = glob.glob(args.file_dir+'post_*.dat')
        if len(pfile_list)==0:
            sys.exit("No posterior files in file_dir (%s)"%args.file_dir)
        else:
            pfile_list = sorted(pfile_list)
            print("%d posterior files detected"%(len(pfile_list)))
                
    elif args.file is not None: # run for only one network
        # check existence
        if os.path.exists(args.file):
            pfile_list.append(args.file)
        else:
            sys.exit("No posterior file exits in the path")
    else:
        sys.exit("No file or file_dir specified")
    
    
    # Run
    t_start = time()
    print("Fit %d models"%len(pfile_list))
    
    for i_model, pfile in enumerate(pfile_list):
    
        # Read posterior from each file + filename setup
        
        pfile_name = os.path.basename(pfile).replace('post_','').replace('.dat','') # post_XXXX.dat
        
        # filenames to be saved
        fig_file = save_dir + 'figs/fitfig_'+pfile_name+'.png'
        param_file = save_dir +'fitparam_'+pfile_name+'.txt'
        
        if renewal==False and os.path.exists(fig_file) and os.path.exists(param_file).is_file():
            print('[PASS] Already done model %s'%pfile_name)
            continue
        
        data = ascii.read(pfile, delimiter='\t', format='commented_header')
        
        if params_to_fit is None:
            params_to_fit = data.colnames
        if params_to_plot is None:
            params_to_plot = data.colnames
        
        #--------------------- binning setup ----------------------------
        nbin=60
        nbin = np.sqrt(len(data)).astype(int)
        #  use  Freedman-Diaconis rule: https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
        use_fdrule = False
        nbin_min = 20 #np.sqrt(len(data)).astype(int)
        nbin_max = 100;
        if i_model==0:
            if use_fdrule:
                print('Use Freedman-Diaconis rule (min:%d, max:%d)'%(nbin_min, nbin_max))
            else:
                print('Use fixed nbin=sqrt(N_data): %d'%nbin)
        
    
        print('Start model: ',i_model)
        fit_info = []
        hist_dic = {}
        nfit_param = 1
        
        for i_param, param in enumerate(params_to_fit):
            
            post = data[param]
            
            if use_fdrule:
                # iqr = np.subtract(*np.percentile(post, [75, 25]))
                # h = 2*iqr*(len(post))**(-1./3)
                # bins = np.round( (np.nanmax(post)-np.nanmin(post))/h  ).astype(int)
                bins = calculate_fd_nbin(post)
                if bins < nbin_min: bins=nbin_min
                if bins > nbin_max: bins=nbin_max
                # bin_statistics.append(bins)
                # print('%s nbin: %d'%(param, bins))
                
            else:
                bins = nbin
                
            yhis, _xhis = np.histogram(post, bins=bins, density=True)
            xhis = 0.5*(_xhis[1:]+_xhis[:-1])
            try: 
                
                popt, pcov = fit_hist(yhis, xhis, N_max=6, initial_offset = 0.0, fresi_success = 0.03, fresi_add=0.05)
                
                fit_info.append(popt)
                if len(popt)>nfit_param:
                    nfit_param = len(popt)
        
            except:
                # change the binning method
                if use_fdrule:
                    bins = nbin
                else:
                    bins = calculate_fd_nbin(post)
                    if bins < nbin_min: bins=nbin_min
                    if bins > nbin_max: bins=nbin_max
                
                yhis, _xhis = np.histogram(post, bins=bins, density=True)
                xhis = 0.5*(_xhis[1:]+_xhis[:-1])
                
                try: 
                    
                    popt, pcov = fit_hist(yhis, xhis, N_max=6, initial_offset = 0.0, fresi_success = 0.03, fresi_add=0.05)
                            
                    fit_info.append(popt)
                    if len(popt)>nfit_param:
                        nfit_param = len(popt)
    
                except:         
                    fit_info.append(np.array([np.nan]))
                    # ax.step(xhis, yhis, where='mid', lw=1.5, label='raw')
                    print('\t fitting fails in %s'%param)
                    # failed_dic[param].append(i_model)
                    
            hist_dic[param] = (xhis, yhis)
                    
        # figrue
        if plot_figure or save_figure:
            
            nrow = np.sqrt(len(params_to_plot)).astype(int)
            if len(params_to_plot)%nrow ==0:
                ncol = len(params_to_plot)//nrow
            else:
                ncol = len(params_to_plot)//nrow + 1
                
            fig, axis = plt.subplots(nrow, ncol,figsize=[3.5*ncol, 3.2*nrow], tight_layout=True)
            axis = axis.ravel()
            
            for i_param, param in enumerate(params_to_plot):
                flag_fit = False
                if param in hist_dic.keys():
                    xhis, yhis = hist_dic[param]
                    popt = fit_info[params_to_fit.index(param)]
                    if np.sum(np.isfinite(popt))==len(popt):
                        flag_fit=True
                        N_components = len(popt)//3
                else:
                    yhis, _xhis = np.histogram(post, bins=bins, density=True)
                    xhis = 0.5*(_xhis[1:]+_xhis[:-1])
                    
                ax = axis[i_param]
                ax.step(xhis, yhis, where='mid', lw=1.5, label='raw')
                if flag_fit:
                    ax.plot(xhis, multi_gauss(xhis, popt), lw=1.5, label="fit")
                    ax.plot(xhis, yhis - multi_gauss(xhis, popt), label="residual")
                    if N_components > 1:
                        mode_handles = []
                        for n in range(N_components):
                            l=ax.plot(xhis, multi_gauss(xhis, np.append(popt[0],popt[1+n*3:1+n*3+3])), ls='--', lw=1, label='mode %d'%(n+1))
                            ax.fill_between(xhis, multi_gauss(xhis, np.append(popt[0],popt[1+n*3:1+n*3+3])), alpha=0.1, hatch=hatch_list[n], color=l[0].get_color(), )
                            mode_handles.append(l[0])
                
                if i_param==0:
                    ax.legend()
                elif flag_fit:
                    if N_components>1:
                        ax.legend(handles=mode_handles)
                      
                ax.set_xlabel(title_dic[param], size=xlabelsize)
                ax.set_ylabel('Probability density', size=ylabelsize)
                ax.yaxis.set_major_locator(ticker.MaxNLocator(2))
                ax.tick_params(axis='y', labelsize=yticksize)
                ax.tick_params(axis='x', labelsize=xticksize)
        
    
        if save_figure:
            fig.savefig(fig_file, dpi=300)
        plt.close()
        
        
        fit_params = np.zeros(shape=(nfit_param, len(fit_info)))+np.nan
        for i, p in enumerate(fit_info):
            fit_params[:len(p), i] = p
        fit_params=Table(fit_params, names=params_to_fit)
        ascii.write(fit_params, param_file, format='commented_header', delimiter='\t')
        
        print('Finished model: ',pfile_name)
            
        
        gc.collect()
        
    t_end = time()
    print('Finished (%.2f min)'%( (t_end-t_start)/60. ))   
    
    # print('Average number of bins: %d'%( np.mean(np.array(bin_statistics) )  ))
        
        
    
        
        
        
        
        
        
        
        
        
        
        
        