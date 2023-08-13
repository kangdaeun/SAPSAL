#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:49:36 2022

@author: daeun

train tools
"""

import numpy as np

# Convergence 
CONV_CUT = 1e-5
N_CONV_CHECK = 20

CONV_CUT_ROUGH = 2e-3
N_CONV_CHECK_ROUGH = 100
N_CONV_CHECK_ROUGH_START = 200

# Divergence
DIVG_CHUNK_SIZE = 7
N_DIVG_CHECK = 30
DIVG_CRI = 0


def check_divergence(loss_array, chunk_size=DIVG_CHUNK_SIZE, n_divg_check=N_DIVG_CHECK, divg_cri=DIVG_CRI):
    """
    Divergence condition:
        1) loss gradually increases
        2) converged but negative log likelihood is positive
        3) keep negative log likelihood positive values

    Parameters
    ----------
    loss_array : TYPE
        DESCRIPTION.
    chunk_size : TYPE, optional
        DESCRIPTION. The default is DIVG_CHUNK_SIZE.
    n_divg_check : TYPE, optional
        DESCRIPTION. The default is N_DIVG_CHECK.
    divg_cri : TYPE, optional
        DESCRIPTION. The default is DIVG_CRI.

    Returns
    -------
    bool
        DESCRIPTION.

    """
    # 1) gradual incrase
    chunk_med = np.array([np.median(loss_array[i:i+chunk_size]) for i in range(0, len(loss_array), chunk_size)])
    roi_divg = chunk_med[1:]-chunk_med[:-1] > (divg_cri/n_divg_check*chunk_size + 5e-3)
    roi_divg = np.append([False], roi_divg)
    n_check = np.ceil(n_divg_check/chunk_size).astype(int)
    
   
    
    
    divergence = None
    # nan -> divergence
    if np.isfinite(loss_array[-1])==False:
        return True
    # positivie value for neg_loglikelihood -> divergence
    if np.median(loss_array[-5:])>0:
        return True
    
    if np.sum(roi_divg[-n_check:]) == n_check:
        divergence = True
    elif np.sum(loss_array[-n_divg_check:] >= 0)==n_divg_check:  # 2) , 3)
        divergence = True
    else:
        divergence = False
        
    return divergence

def check_convergence(loss_array, conv_cut=CONV_CUT, n_conv_check=N_CONV_CHECK):
    roi_conv = abs(loss_array[1:]/loss_array[:-1] -1 ) < conv_cut
    roi_conv = np.append([False], roi_conv)
    
    if np.isfinite(loss_array[-1])==False:
        return False
    if np.median(loss_array[-5:])>0:
        return False
    
    if np.sum(roi_conv[-n_conv_check:])==n_conv_check:
        return True
    elif len(loss_array) > N_CONV_CHECK_ROUGH + N_CONV_CHECK_ROUGH_START: # rough second check
        roi_conv_rough = abs(loss_array[1:]/loss_array[:-1] -1) < CONV_CUT_ROUGH
        roi_conv_rough = np.append([False], roi_conv_rough)  
        if np.sum(roi_conv_rough[-N_CONV_CHECK_ROUGH:])==N_CONV_CHECK_ROUGH:
            return True
        else:
            return False
    else:
        return False
    

def rewrite_config_element(config_file,  new_config_file, param_to_change, value_to_change,):
    with open(config_file, 'r') as f1:
        config_components = f1.read().split('\n')
        
    roi_comment = np.array(["#" in comp or '"""' in comp or "'''" in comp for comp in config_components])  
    
    i_comp = np.where(np.array([ param_to_change in comp for comp in config_components]) *np.invert(roi_comment))[0][0]

    if isinstance(value_to_change, str):
        new_comp = "%s = '%s'"%(param_to_change, value_to_change) # Else the str will miss the quotation marks
    else:
        new_comp = "%s = %s" %(param_to_change, value_to_change)   
    config_components[i_comp] = config_components[i_comp].replace(config_components[i_comp], new_comp)

    with open(new_config_file, 'w') as f1:
        f1.write('\n'.join(config_components))