#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 18:33:31 2022

@author: daeun

Tools for run_hypertrain

"""
import numpy as np
import re

def rand_in(a, b):
    '''
    Returns a random uniform real number between a [float] and b [float]
    '''
    return a + np.random.rand() * (b-a)

def rand_in_log10(a, b):
    '''Returns a random uniform real number between 10**a and 10**b
        a [float], b[float] '''
    return 10 ** rand_in(a, b)

def rand_in_log2(a, b):
    '''Returns a random power of 2 between 2**a and 2**b
        a[int], b[int] '''
    return 2 ** np.random.randint(a, b+1)

def rand_in_discrete(options):
    '''Returns a random element from a list of discrete options
        options [list] '''
    return options[np.random.randint(len(options))]

def rand_bool():
    '''Returns a random bool'''
    return bool(np.random.randint(2))


    
def find_str_names(c, config_file, dim_max=20):

    n_x = len(c.x_names); n_y = len(c.y_names)
    if n_x <= dim_max and n_y <= dim_max:
        return None, None
    else:
        with open(config_file, 'r') as f:
            text = f.read()
    
        # Remove comments to avoid confusions
        text = re.sub(r'("""|\'\'\')(.*?)\1', '', text, flags=re.DOTALL)
    
        lines = text.split('\n')
        config_components = []
        for line in lines:
            stripped = line.split('#')[0].rstrip()  # # 이후 삭제
            if stripped:  # 빈 줄은 제외
                config_components.append(stripped)
        
        if n_x > dim_max:
            i_comp = np.where( np.array([("x_names" in comp) and ("#" not in comp) for comp in config_components]))[0][0]
            str_x_names = config_components[i_comp].split("=")[-1].strip()
        else:
            str_x_names = None
            
        if n_y > dim_max:
            i_comp = np.where( np.array([("y_names" in comp) and ("#" not in comp) for comp in config_components]))[0][0]
            str_y_names = config_components[i_comp].split("=")[-1].strip()
        else:
            str_y_names = None
            
        return str_x_names, str_y_names
            
    
    
def find_str_flag_names(c, config_file, dim_max=5):
    
    if c.use_flag == True:
        str_f_names = []
        for i_flag, flag_name in enumerate(c.flag_names):
            val = c.flag_dic[flag_name]
    else:
        return None
    
    n_x = len(c.x_names); n_y = len(c.y_names)
    if n_x <= dim_max and n_y <= dim_max:
        return None, None
    else:
        with open(config_file, 'r') as f1:
            config_components = f1.read().split('\n')
        
        if n_x > dim_max:
            i_comp = np.where( np.array([("x_names" in comp) and ("#" not in comp) for comp in config_components]))[0][0]
            str_x_names = config_components[i_comp].split("=")[-1].strip()
        else:
            str_x_names = None
            
        if n_y > dim_max:
            i_comp = np.where( np.array([("y_names" in comp) and ("#" not in comp) for comp in config_components]))[0][0]
            str_y_names = config_components[i_comp].split("=")[-1].strip()
        else:
            str_y_names = None
            
        return str_x_names, str_y_names    
    
    
def randomize_config(c_ref, search_parameters, output_filename,
                     device, #n_epochs_max,
                     adjust_gamma_n_epoch=False,
                     auto_n_epoch=False,
                     hs_id=None):
    '''
    Randomizes the elements of the config dictionary config_dict specified
    in search_parameters
    -------------------
    c_ref [config class]: reference config
    search_paramters [dict]: parameters to randomize and corresponding functions
    output_filename: [str]: name of the output file ends with ~.pt
    device [str]
    n_epochs_max [int]
    (currently not used)
    auto_n_epoch [bool]
    hs_id [str]
    '''    
    if c_ref.network_model is not None:
        c_ref.network_model = None
        
    c = c_ref.copy()
    # Set output filename (filename contains relative path)
    c.filename = output_filename 
    
    # Update output directory and device
    c.device = device
    
    # Randomize settings specified in search parameters
    for key in search_parameters.keys():
        func_key, qwargs_key = search_parameters[key]
        setattr(c, key, func_key(**qwargs_key) )
        
        
    # If using domain adaption and da_disc_set_optim = False, set da_disc setting same as main network
    if c.domain_adaptation:
        if c.da_disc_set_optim != True:
            c.update_da_optim()
        
        
    # Set all verbosity switches to False 
    # c.checkpoint_save = False
    c.progress_bar = False
    c.live_visualization = False
    
    if adjust_gamma_n_epoch:
        if c.gamma >= 0.7 and c.n_epochs < 300: 
            c.n_epochs = 300
        elif c.gamma >= 0.5 and c.n_epochs < 200:
            c.n_epochs = 200
        
    # Limit n_epochs ( <= N_EPOCHS_MAX ) : not used anymore
    # if c.n_epochs > n_epochs_max: 
    #     c.n_epochs = n_epochs_max
    
    return c
    

