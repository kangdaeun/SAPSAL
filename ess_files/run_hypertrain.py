#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 15:00:50 2022

@author: daeun


hyperparam 랜덤으로 바꿔가며 여러 네트워크를 훈련

[To do]
- 최종 그림들 옮기는 것
- outputdir도 설정할 수 있으면?


1) hyperparam 조건 셋팅
    - 파라미터마다 랜덤하게 사용할 함수 선택 필요
    - 고정되는 정보들이 있다: 
        x_names, y_names, tablename, filename의 코드, ouput/
        Model, 몇몇 하이퍼파라미터,
    - 고정정보만을 담은 config를 넣어서 시작
        - 고정정보에 들어있는 것들을 랜덤하게 하지 않도록 (서치 딕셔너리에서 제외)
    - Model에 따라 관련 안되어있는 하이퍼파라미터도 제외
For 
2) config 작성, 훈련, 기본적인 eval
- config는 for 돌때마다 랜덤으로 값을 산출하여
- 훈련 (함수사용):
    - 수렴/발산을 판단할 수 있어야
    - 필요에 따라 에폭을 연장할 수 있어야.
- eval :
    - 선택에 필요한 지표들을 계산하는 역할
endFor
3) 판단. 혹은 판단 지표



parser.add_argument('config_file', help="Run with specified config file as basis.")
parser.add_argument('-s','--suffix', required=False, default=None, help="Output suffix")
parser.add_argument('-o','--outputdir', required=False, default=None, help="Output directory")
parser.add_argument('-r','--resume', required=False, default=None, help="Resume Hyperparameter search or not (T/F)")
    
"""

import numpy as np
import os, glob, sys
import pandas as pd
import subprocess
import multiprocessing
from argparse import ArgumentParser
# import random
# from itertools import repeat
from time import time
from time import sleep

from sapsal.cINN_config import read_config_from_file
# from sapsal.data_loader import *
# from sapsal.execute import train_network as train_normal_network

import sapsal.tools.hs_tools as tools
import sapsal.tools.test_tools as test_tools #combine_multiple_evaluations


###########
## Setup ##
###########

N_RAND = 40                         # Number of random configs to generate
SEED = 25081136                       # Random seed for config randomisation # 25011132               

N_PROCESSES = 1           # Number of multiple processes


OUTPUT_DIR = "HP_search_random/"    # Path to output directory
OUTPUT_SUFFIX = "HP_NET"           # Optional suffix for all output files (can change by option argument)

# for Slurm jobs
NUM_CPUS = None
if "SLURM_CPUS_PER_TASK" in os.environ:
    NUM_CPUS = int(os.environ["SLURM_CPUS_PER_TASK"])
elif "SLURM_JOB_CPUS_PER_NODE" in os.environ:
    NUM_CPUS = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])

if NUM_CPUS:
    print("NUM_CPUS:",NUM_CPUS)
    os.environ["OMP_NUM_THREADS"] = str(NUM_CPUS)
    os.environ["NUMEXPR_MAX_THREADS"] = str(NUM_CPUS)

DEVICE = 'cuda'              # Device to run search on, either 'cuda' or 'cpu'
# These GPU setup is not used in this code.
GPU_MAX_LOAD = 0.1          # Maximum compute load of GPU allowed in automated selection
GPU_MAX_MEMORY = 0.1         # Maximum memory load of GPU allowed in automated selection
GPU_WAIT_S = 600             # Time (s) to wait between tries to find a free GPU if none was found
GPU_ATTEMPTS = 10            # Number of times to retry finding a GPU if none was found
GPU_EXCLUDE_IDS = [] # List of GPU IDs that are to be ignored when trying to find a free GPU, leave as empty list if none

# N_EPOCHS_MAX  = 500        # Maximum n_epochs for training: this is set in run_train.py

# AUTO_N_EPOCHS = False    # Switch to automatically set n_epoch if batch_size and n_its_per_epoch are randomised
VERBOSE = True           # Switch for progress messages
RESUME = False          # Switch to resume a previous hyperparameter search
SAVE_PLOT_COPY = True  # copy plots 




# Dictionary with the config parameters to randomize, the way to randomize them
# and parameters to the randomisation function
SEARCH_PARAMETERS = {
    "gamma" : (tools.rand_in, {"a":0.1, "b":0.7}), # usually 0.1~0.8
    "adam_betas" : (tools.rand_in_discrete, {"options":[(0.8, 0.8), (0.9, 0.9), (0.5, 0.8), (0.5, 0.9)]}),
    "lr_init" : (tools.rand_in_log10, {"a":-4, "b":-2}), # -4 ~-2
    "l2_weight_reg" : (tools.rand_in_log10, {"a":-4.3, "b":-2.0}), # (-4.3, -2.5)
    "meta_epoch" : (np.random.randint, {"low":5, "high":15+1}), # 5~15
    "n_blocks" : (np.random.randint, {"low":8, "high":16+1}), # noise-net: 24+1, normal-net:16+1
    "internal_layer": (np.random.randint, {"low":3, "high":6+1}),
    # # "batch_size" : (tools.rand_in_log2, {"a":8 ,"b":9}),
    "internal_width" : (tools.rand_in_log2, {"a":8 ,"b":10}), # usually fixed to 256
    "feature_layer": (np.random.randint, {"low":3, "high":6+1}), # usually fixed to 3 (for Noise 3~9+1) (for Normal 3~5+1)
    "feature_width" : (tools.rand_in_log2, {"a":8 ,"b":10}), # usually fixed to 512
    "y_dim_features" : (tools.rand_in_log2, {"a":7, "b":9}),
    # "n_its_per_epoch" : (tools.rand_in_log2, {"a":8, "b":10}),
#    # "fcl_internal_size" : (tools.rand_in_log2, {"a":7, "b":11}), 
#    "scale_data" : (tools.rand_bool, {}),
#    # "do_rev" : (tools.rand_bool, {}),
#    "use_feature_net" : (tools.rand_bool, {}),
     # "da_disc_train_step":(np.random.randint, {"low":1, "high":2+1}), 
     # "da_disc_train_step":(tools.rand_in_discrete, {"options":[1, None]}),
     # "lambda_adv": (tools.rand_in_discrete, {"options":[1, 1.5, 2]}),
     # "da_disc_gamma" : (tools.rand_in, {"a":0.1, "b":0.6}),
     # "da_disc_lr_init" : (tools.rand_in_log10, {"a":-6, "b":-4}),
     # "da_disc_l2_weight_reg" : (tools.rand_in_log10, {"a":-4, "b":-2.0}), # (-4.3, -2.5)
     # # "da_disc_adam_betas" : (tools.rand_in_discrete, {"options":[(0.8, 0.8), (0.9, 0.9), (0.5, 0.8), (0.5, 0.9)]}),
     #  "da_disc_layer": (np.random.randint, {"low":3, "high":6+1}),
     # # "da_disc_width" : (tools.rand_in_log2, {"a":8 ,"b":10}), # 
     

    "seed_weight_init" : (np.random.randint, {"low":1, "high":241542})
    }

#%%

def change_suffix(suffix):
    global OUTPUT_SUFFIX # change global variable
    OUTPUT_SUFFIX = suffix
    
def change_outputdir(outputdir):
    global OUTPUT_DIR # change global variable
    if outputdir[-1] != '/':
        outputdir += '/'
    OUTPUT_DIR = outputdir
    
def change_gpu_exclude_ids(num, add=None, remove=None):
    global GPU_EXCLUDE_IDS # change global variable
    listOfGlobals = globals()
    # print(GPU_EXCLUDE_IDS)
    if add:
        listOfGlobals['GPU_EXCLUDE_IDS'].append(num)
    if remove:
        try:
            listOfGlobals['GPU_EXCLUDE_IDS'].remove(num)
        except:
            print("num not in GPU_EXCLUDE_IDS")

    
def update_config_table(config, n):
    
    config_df_path = OUTPUT_DIR + "Config_table_%s.csv" % OUTPUT_SUFFIX
    config_data = [['%d'%n, config.config_file]+[getattr(config, k) for k in SEARCH_PARAMETERS.keys()]]
    
    if os.path.exists(config_df_path):
        config_df = pd.read_csv(config_df_path)
        config_df_columns = config_df.columns
        config_df = pd.concat([config_df, pd.DataFrame(data=config_data, columns=config_df_columns)], 
                              axis=0, ignore_index=True)
    else:
        config_df_columns = ['ID', 'FILE']+list(SEARCH_PARAMETERS.keys())
        config_df = pd.DataFrame(data=config_data, columns=config_df_columns)
        
    config_df.to_csv(config_df_path, index=False)
    
def read_config_table():
    try:
        config_df_path = OUTPUT_DIR + "Config_table_%s.csv" % OUTPUT_SUFFIX
        return pd.read_csv(config_df_path)
    except:
        print("No config table to read")
        return False
    
def remove_config_table():
    try:
        os.system('rm '+OUTPUT_DIR + "Config_table_%s.csv" % OUTPUT_SUFFIX)
    except Exception as e:   
        print('(remove_config_table):', e)
            

def make_random_config(c_ref, n):
    
    # Setup new config path and output directory
    config_path_n = OUTPUT_DIR + "configs/c_%s_%02d.py" % (OUTPUT_SUFFIX, n)
    output_filename_n = OUTPUT_DIR + "model_%s_%02d/%s_%02d.pt" % (OUTPUT_SUFFIX, n,
                                                           OUTPUT_SUFFIX, n)
    
    if VERBOSE:
        print("Randomising config %d"%n)
        
    # Randomly make one config (class)
    config = tools.randomize_config(c_ref, SEARCH_PARAMETERS,
                                output_filename_n, DEVICE, #N_EPOCHS_MAX,
                                adjust_gamma_n_epoch=True)                         
    
    # write and save config
    str_x_names, str_y_names = tools.find_str_names(c_ref, c_ref.config_file, dim_max=20)
    config.save_config(config_file = config_path_n,
                         config_comment = "Hypermaramter search: %i"%n,
                          str_x_names = str_x_names, str_y_names=str_y_names, verbose=False )
    config.config_file = config_path_n
    # update config table
    update_config_table(config, n)

    return config
    
#%%   
def set_device(config):
    # Find a free GPU to run on if DEVICE == cuda
    if "cuda" in config.device:
        # Get the first available GPU device ID
        DEVICE_ID_LIST = GPUtil.getFirstAvailable(maxLoad=GPU_MAX_LOAD,
                                                  maxMemory=GPU_MAX_MEMORY,
                                                  attempts=GPU_ATTEMPTS,
                                                  interval=GPU_WAIT_S,
                                                  excludeID=GPU_EXCLUDE_IDS,
                                                  verbose=VERBOSE)
        device_id = DEVICE_ID_LIST[0]
        config.device = 'cuda:{:d}'.format(device_id)
        change_gpu_exclude_ids(int(device_id), add=True)
        # print("In set_device:",config.device)
    
    return device_id

# not actually used now
def check_device(device_id):
    # check current is still available
    exclude_id = [i for i in GPU_EXCLUDE_IDS]
    try:
        exclude_id.remove(device_id)
    except:
        pass
    bool_available = GPUtil.getAvailability(GPUtil.getGPUs(), maxLoad=GPU_MAX_LOAD, maxMemory=GPU_MAX_MEMORY,
                           excludeID=exclude_id)
    return bool( bool_available[int(device_id)] )

#%%
def make_log(log_dir, model_name, run_name):
    logfile = log_dir + model_name + "_{}.log".format(run_name)
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    return open(logfile, "a")
    
def train_network(config, logfile=None):
    
    """
    Parameters
    ----------
    config : TYPE
        DESCRIPTION.
    logfile : str, optional
        If you are using a special log. Full path of the log. The default is None.

    """
    if logfile is None:
        f_train_log = make_log( os.path.dirname(config.filename)+'/',
                                os.path.basename(config.filename),
                                'train' )
        close_log = True
    else: 
        close_log = False
        
    if VERBOSE:
        print("Starting training: %s"%(os.path.basename(config.config_file)) )
        
    
        # train_normal_network(config, data=astro, verbose=VERBOSE)
        # args = ['python','ess_files/run_train.py', config.config_file, config.device]
        # if 'cuda' in config.device:
        #     if not check_device(int(config.device.replace('cuda:',''))):
    args = ['python','-u','ess_files/run_train.py', config.config_file, '-r', '--log', 'False']
    proc = subprocess.Popen(args,
                            stdout=f_train_log, stderr=f_train_log, start_new_session=True).wait()
    
    if close_log:
        f_train_log.close()
    
 
    
def evaluate_network(config, logfile=None):
    """
    Evaluations: latent, eval (calibration, MAP, True vs Post)

    Parameters
    ----------
    config : TYPE
        DESCRIPTION.
    logfile : str, optional
        If you are using a special log. Full path of the log. The default is None.

    Returns
    -------
    None.

    """

    if logfile is None:
        f_eval_log = make_log( os.path.dirname(config.filename)+'/',
                                os.path.basename(config.filename),
                                'eval' )
        close_log = True
    else: 
        close_log = False
    
    if VERBOSE:
        print("Starting evaluations: %s"%(os.path.basename(config.config_file)) )    
       
        
        # args = ['python','ess_files/eval_network.py', config.config_file, config.device]
        # if 'cuda' in config.device:
        #     if not check_device(int(config.device.replace('cuda:',''))):
    args = ['python','-u','ess_files/eval_network.py', config.config_file, '--log', 'False']

    # proc1 = subprocess.Popen(['python','ess_files/plot_z_pdf.py', config.config_file, config.device], 
    #                         stdout=f_eval_log, stderr=f_eval_log, start_new_session=True).wait()
    proc2 = subprocess.Popen(args,
                            stdout=f_eval_log, stderr=f_eval_log, start_new_session=True).wait()
        
        
    if close_log:
        f_eval_log.close()
 
#%%
# not actually used
def run_all(c_ref, n, **kwargs): 
    
    """
    Parameters
    ----------
    c_ref : config class
        reference config.
    n : int
        index of the config (n th config).
        
    kwargs :
        verbose
    -------
    Set for one random config.

    """
    
    config = make_random_config(c_ref, n)
    device_id = set_device(config)
    sleep(15)
    train_network(config, logfile=None)
    evaluate_network(config, logfile=None)
    change_gpu_exclude_ids(device_id, remove=True)
    
    
def prepare_config(c_ref, n):
    """
    Prepare individual config one by one

    Parameters
    ----------
    c_ref : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.

    Returns
    -------
    config : TYPE
        DESCRIPTION.

    """    
    config = make_random_config(c_ref, n)
    print(GPU_EXCLUDE_IDS)
    if 'cuda' in config.device:
        device_id = set_device(config) # already include to exclude
        # print(GPU_EXCLUDE_IDS)
    return config

# Currently Used
def train_and_eval(config):
    """
    Set of training and evaluation. If RESUME, check status and run
    IF not RESUME, always restart

    Parameters
    ----------
    config : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # if 'cuda' in config.device:
    #     set_device(config)
    
    # If RESUME, you need to check if you already done the training / not RESUME -> train newly
    if RESUME==True and test_tools.check_train_status(config):
        print("Alreay done training (%s)"%os.path.basename(config.config_file) )
        new_train = False
    else:
        while test_tools.check_training_status(config) is None:
            train_network(config, logfile=None)
        new_train = True
    
    if (RESUME==True)*(test_tools.check_eval_status(config))*(new_train==False):
        print("Alreay done evaluation (%s)"%os.path.basename(config.config_file) )
    else:    
        # check training status (converged/diverged): if diverged (-1) then do not run evaluation (None: No file)
        training_status = test_tools.check_training_status(config)
        if training_status == -1:
            print("Network diverged. Pass evaluation (%s)"%os.path.basename(config.config_file))
        
        else:
            test_tools.clean_evalfiles(config)
            while test_tools.check_eval_status(config)==False:
                evaluate_network(config, logfile=None)
    
    # Release GPU
    if 'cuda' in config.device:
        change_gpu_exclude_ids(int(config.device.replace('cuda:','')), remove=True)
    print("\t Finished %s\n"%os.path.basename(config.config_file) )
    
#%% Main
##########
## MAIN ##
##########

if __name__=='__main__':
    
    # Parse optional command line arguments
    parser = ArgumentParser()
    # parser.add_argument("-c", "--config", dest="config_file", default='./config.py',
    #                     help="Run with specified config file as basis.")
    
    parser.add_argument('config_file', help="Run with specified config file as basis.")
    parser.add_argument('-s','--suffix', required=False, default=None, help="Output suffix")
    parser.add_argument('-o','--outputdir', required=False, default=None, help="Output directory")
    parser.add_argument('-r','--resume', required=False, default=None, help="Resume Hyperparameter search or not (T/F)")
    parser.add_argument('-n','--n_start', required=False, default=None, help="Start config from this numner (if resume=T)")
    
   
    args = parser.parse_args()
    
    if args.suffix is not None:
        change_suffix(str(args.suffix) )
    if args.outputdir is not None:
        change_outputdir(str(args.outputdir) )
    if args.resume is not None:
        if args.resume == 'False' or args.resume=='0':
            RESUME = False
        if args.resume =='True' or args.resume=='1':
            RESUME = True
            
    if RESUME==True and args.n_start is not None:
        N_start = int(args.n_start)
    else:
        N_start = 0

    # Import default config
    config_file = args.config_file
    c_ref = read_config_from_file(config_file, verbose=False)
    c_ref.device = DEVICE
    
   # Create output directory if it does not exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Create subfolder for the configs if it does not exist
    if not os.path.exists(OUTPUT_DIR + "configs/"):
        os.makedirs(OUTPUT_DIR + "configs/")

    # Create the optional folder for copies of the plots
    if SAVE_PLOT_COPY and not os.path.exists(OUTPUT_DIR + "plots/"):
        os.makedirs(OUTPUT_DIR + "plots/")

    # Import GPUtil if the search is to be run using GPUs
    if DEVICE == "cuda":
        import GPUtil
    

    # 1. Make all config files 
    # IF RESUME, check if you already prepared all configs
    # or need to generate additional config files and update csv
    # N_start = 0
    np.random.seed(SEED)
    
    if RESUME:
        # check configfiles saved
        n_saved = len(glob.glob(OUTPUT_DIR + "configs/c_*.py"))
        # check updated configs
        config_df = read_config_table()
        if config_df is False:
            n_updated = 0
        else:
            n_updated = config_df['ID'].values[-1]+1
        
        if n_saved != n_updated:
            # First, reupdate all existing configfiles 
            remove_config_table()
            for n, file in enumerate( sorted(glob.glob(OUTPUT_DIR + "configs/c_*.py")) ):
                c_last = read_config_from_file(file, verbose=False)
                update_config_table(c_last, n)
                n_updated = n_saved
        
        # Make additional config files if needed
        if n_updated < N_RAND:
            print("[RESUME] Make dditional config files")
            # change seed 
            np.random.seed(SEED+n_updated)
            for n in range(n_updated, N_RAND):
                c_last = make_random_config(c_ref, n)
            print("Saved all config files.\n")
        else:
            print("[RESUME] Config files are already prepared")
    else:
        # Totally New Run
        # Remove config csv and model directories (in case they exist)
        # remove all model~/ directories
        remove_config_table()
        os.system('rm -rf %s'%(OUTPUT_DIR + "model_*/"))
        os.system('rm -rf %s'%(OUTPUT_DIR + "configs/*"))
        
        print("Make config files")
        for n in range(N_start, N_RAND):
            c_last = make_random_config(c_ref, n)
        print("Saved all config files.\n")
        
                
    # 2. Execute Loop : Train + Evaluation
    sys.exit()
    t_start = time()
    t_mid = t_start
        
    config_df = read_config_table()
    config_files = config_df['FILE'].values

    print("Use %d processes"%N_PROCESSES)
    print()
    pool = multiprocessing.Pool(N_PROCESSES)
    # pool.starmap(run_train_to_eval, zip(repeat(c_ref), range(N_start, N_RAND)))
    
    if 'cuda' in DEVICE:
        for i in range(5):
            change_gpu_exclude_ids(0, add=True)
            change_gpu_exclude_ids(0, remove=True)
     
    for n in range(N_start, N_RAND):
        
        # 1.Read config file
        config = read_config_from_file(config_files[n], verbose=False)
       
        
        if RESUME==True and test_tools.check_train_status(config) and test_tools.check_eval_status(config):
            print("Alreay done (%s)"%os.path.basename(config.config_file) )
        else:
            # 2. Set device 
            # device_id = set_device(config)
            pool.apply(print, args=[os.path.basename(config.config_file)])
            # device_id = pool.apply(set_device, args=[config])
            # config.device = 'cuda:{:d}'.format(device_id)
            # print('After set_device:',config.device)
            
            # 3. Train and evaluate
            pool.apply_async(train_and_eval, args=[config] )
            
            
            pool.apply(sleep, args=[5])
        
        if time()-t_mid > 30*60:
            t_mid = time()
            print("\n %.1f min (%.1f h) passed \n"%( (t_mid-t_start)/60, (t_mid-t_start)/3600. ) ) 
    
    pool.close()
    pool.join()
    
    
    # 3. collect all results and make final results
    config_list = []
    # for cfile in sorted(glob.glob(OUTPUT_DIR + "configs/c_*.py")):
    for cfile in config_files:
        config_list.append(read_config_from_file(cfile))
        
    eval_df_filename = OUTPUT_DIR + "Eval_table_%s.csv" % OUTPUT_SUFFIX
    eval_df = test_tools.combine_multiple_evaluations(config_list, filename=eval_df_filename, 
                                 sep=',', return_output=True)
    
    # Make full dataframe (config + evaluation)
    final_df = pd.concat( [config_df, eval_df],  axis=1, ignore_index=False)
    final_df_filename = OUTPUT_DIR + "HP_search_result_%s.csv" % OUTPUT_SUFFIX
    final_df.to_csv(final_df_filename, index=False)
    
    # copy plots and make individual directories if SAVE_PLOT_COPY=True
    if SAVE_PLOT_COPY:
        print("Copy test plots")
        # loss plot, z_corr, z_cov, z_qq, TvT, TvP_MAP, alib
        plotdir = OUTPUT_DIR + "plots/"
        for key, suffix in test_tools.testfigure_suffix.items():
            ori = OUTPUT_DIR + "model_%s_*/*.pt"%OUTPUT_SUFFIX + suffix
            cpy = plotdir + key + '/'
            if not os.path.exists(cpy):
                os.makedirs(cpy)
            os.system('cp {} {}'.format(ori, cpy))
            print("Copied %s plot"%key)
    
    
   
    # Status message
    

    # 1: Make one randomized config file
    # 2. Collect config information
    # 3. Train
    # 4. Evaluation
    # 5. Save/append necessary info     
    # 6. combine evaluation results


    t_end = time()
    print("Finished Hyperparameter search")
    dt = t_end - t_start
    print("Time taken: {:.2f} hour ({:.1f} min)".format( dt/3600, dt/60   ))
    





