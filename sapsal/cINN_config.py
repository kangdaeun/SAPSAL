#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 15:13:26 2021

@author: daeun

cINN network config class

Currently availbale Models for Main Network (2): Glow, AllinOne
- 사용자가 무엇을 선택하냐에 따라 필수적으로 들어가야하는 파라미터가 있고, 이 부분에 대한 설정이 없으면 안됨
 
할지 말지의 flag에 해당하는 것은 무조건 True/False로 (None말고)

Need to set in config by user:
    Output file name
    Database path for training
    x_names, y_names
    fraction of train set and test set (8:2)
    data preprocessing:
        log scale / linear
        clipping
        smoothing 
    Flags for training method (T/F)
        train_noisy_obs [deprecated]
        prenoise_training (Noise-Net training)
        
여기서 디폴트값이라도 해서 파라미터 모듈에 넣어야 하는 것과 없어야 하는 것을 구분       

Attributes:
    data related: 
        x_names : list, name of parameters (X) according to the format of db
        y_names : list, name of observables (Y) according to the format of db
            x_dim = dimension of X to predict
            y_dim_in = dimension of conditioned Y
            > if you have x_names, and y_names you don't have to include x_dim and y_dim_in in your configuration it is updated automatically updated
            > if you change x_names or y_names, than you need to update dimension by 
            >> c.update_dimension()
        
        tablename: path/name of DB to use. 
        expander [deprecated]: expander corresponds to DB. this should be python path/name of python module
        
        proj_dir (not an attribute!) : if you include this in your config file, we regard that (tablename, filename, expander) is relative to projdir
        > we recommend to exclude it or keep this None in your config file
        > update when you use network in other dirctory by
        >> c.set_proj_dir(proj_dir, **kwarg) : you can manually control updating (tablename, filename, expander)
        > or you can set proj_dir when you read config file
        > if you want to set proj_dir for (tablename, filename, expander) as default. just use abspath for (tablename, filename, expander)
        
        
    output network:
        filename = 'output/cINN_code.pt' relative path to projdir or abspath (~.pt) this will be used as name_code for output files
            
    Network model
        model_code = 'ModelAdamGLOW' # name of model class / ModelAdamAllInOne
        
    [DEPRECATED] FrEIA version
        until Kang et al. 2022 we used version 0.1. 
        lots of config files do not contain FrEIA version so default is 0.1
        This affects models and excute
        2023.08.10. FrEIA_ver deprecated. always used FrEIA >= 0.2 attached in cINN_set
        
        
    Training hyperparameters:
        
        # choose which dievice to use for torch Tensor (string)
        # 'cpu', 'cuda', you can specify gpu number by 'cuda:2'
        device = 'cpu'

        # batchsize, use less than 2048
        batch_size = 1024

        # the number of training epochs
        n_epochs = 100

        # the number of additional pre-learning epochs
        # : training epoch starts from -pre_low_lr
        pre_low_lr = 0

        # Max number of batch iteration in one epoch. 
        # Use large enough number > the number of batch in training data = data/batchsize
        # if len(train_loader) > n_its_per_epoch : (too many batches) shutdown
        n_its_per_epoch = 2**16


        # load previous file before start traning
        load_file = '' 

        # chekpoint file is saved when i_epoch%X ==0
        checkpoint_save = True # True/False 이거 관련 train 업데이트 
        checkpoint_save_interval = 120 * 2 
        # this will overwrite checkpoint (making the same name)
        checkpoint_save_overwrite = True 
        
        
    Model optimization hyperparams
        # if gamma is larger than proper value, loss curve has many spikes
        gamma = 0.15

        # initial learning rate
        lr_init = 1e-3 

        # smaller l2 -> less effective
        l2_weight_reg = 1e-5

        # (beta1, beta2) In priciple, beta2->0.999 is good
        # personally do not recommend to reduce betas smaller than 0.8 
        adam_betas = (0.8, 0.8)

        # how often you change the learning rate (1=every epch)
        meta_epoch = 10 

        # Update reverse loss during the training (do_rev = True)
        # reverse loss (lr) = mean( (x_true - x_rev)^2), x_rev is from inverse of forward output(z) 
        # in the inverse process we can add noise to z: latent noise (sigma of gaussian)
        do_rev = False
        latent_noise = 0. # 0.05 used in mnist example, 

    Model construction hyperparams
        # # of coupling blocks of main network
        n_blocks = 8

        # width of subnetwork. subnetwork has 3 layers (fc_constr)
        internal_width = 256 
        # the number of layer in subnetwork. (fc_constr)
        internal_layer = 3

        # [Not sure] initial scaling of training parameters?
        init_scale = 0.03 # used in model (both Glow & All)

        # Not used in any codes
        # fc_dropout = 0.0

        # transfer y to feature (cond_net=feature_net). this is different to subnetwork 
        # [Not sure] why 256?
        y_dim_features = 256 
        feature_width = 512
        feature_layer = 3
        
        # keywords for GLOWCoupling
        exponent_clamping = 2.0 
        use_permutation = True # permuation or not 
        
        # keywords for AllInOneBlock 
        affine_clamping = 2.0
        gin_block = False # same as default (AllInOneBlock)
        global_affine_init = 0.7 #(AllInOneBlock)
        global_affine_type = 'SOFTPLUS' # same as default (AllInOneBlock)
        permute_soft =  True # True = use random orthogonal matrix, False = use an actual hard permutation matrix (subset of orthogonal matrices) (AllInOneBlock)
        reverse_permutation = False # (AllInOneBlock)
        learned_householder_permutation = 0 #(AllInOneBlock)
    
    visualization hyperparams
        loss_names = ['L_train_mn',  'L_test_mn', 'L_train_mdn', 'L_test_mdn'] 
        # loss curve: every 20 epochs + at last
        loss_plot_yrange = [-40, 10]
        loss_plot_xrange = None
        # preview_upscale = 3 # not used????
        # not available right now. not yet modified
        live_visualization = False
        # tqdm bar for batch training
        progress_bar = False 

Excluded param: 
    feature_net_file = '', fc_dropout = 0.0, stop_epoch (not used anymore), decay_by, 
    train_code, viz_code, test_smoothing, preview_upscale
    
    Temporaray test attributes: (setting_param)
        train_noisy_obs = T/F (default: False): enlarge and make noisy obs for training
        n_noise_MC: the number of MC for Y to make noise training 
        noise_fsigma: dictionary like smoothing_sigma but use fraction: dy/y in fraction (not percentage)
        
               
"""

import sys, os
import importlib.util
import numpy as np
import re
import torch # needed when load rescale params
import copy
from .models import *
# from models import *

class cINNConfig():
    
    # these should be set 
    __data_param = {
                    'tablename': None, # or database # abspath
                    'x_names': ['logTeff', 'logG'], # list of parameter names 
                    'y_names': ['l{:d}'.format(i) for i in range(3681)], # list of observation names   
                    # 'expander':None, # module or class # abspath ## expander deprecated. expander is in cINN_set
                    }
    
    __dependent_param = {
                        'x_dim': None,
                        'y_dim_in': None, # updated when y_names is given
                    
                        }
    
    __setting_param = {
                    'filename':  None, # only name. use as ~.pt ~_Loss, ~_..etc
                    'test_frac': 0.2, # 0<frac<1: recommend less than 0.2      
                    'train_smoothing': False,
                    # 'test_smoothing': None,
                    'smoothing_sigma': None, 
                    
#                    'architecture': 'cINN', # DEPREACATED (not used anymore, but leave to read old configs)
                    'model_code': 'ModelAdamGLOW', 
#                    'FrEIA_ver': 0.2, # version of FrEIA (2022.1.10: 0.1 or 0.2) (2023.08.10. DEPRECATED)
                    
                    # New function of cINN-Stellar
                    'normalize_flux': None, # T/F
                    'normalize_total_flux': None, # Methods: T/F
                    'normalize_mean_flux': None,
                    'normalize_f750': None,
                    
                    'use_Hslab_veiling': None, #T/F
                    'use_one_Hslab_model': None, #T/F
                    'slab_grid': None, # path to slab grid file (.csv) # if use_Hslab_veiling=True but use_one_Hslab_model !=True, thatn automatically read slab_grid
                    
                    # Randomizing parameter on the fly
                    'random_parameters': None, # dictionary
                    'additional_kwarg': None, # dictionary : currently used for: Rv

                    # New: Coupling data [TEST]
                    'wavelength_coupling': False, #T/F
                    
                    # Domain adaptaion [TEST]
                    'domain_adaptation': False, # T/F use domain adaption
                    'real_database': None, # path to real database. will be updated with proj_dir
                    'real_frac': 1, # fraction of real data with repect to the batch size: real_frac x batchsize
                    'da_without_discriminator': False,
                    
                    'lambda_adv': 0.1, # 0 < < max?, loss = neg_log_llike + lambda_adv * loss_adv
                    'da_disc_train_step': 1, # how frequently update weight: 1=every batch, if None, it automatically increase from 1
                    'da_mode': 'simple', # genearll ADV, WGAN : this changes the way to calcualte loss
                    'da_label_smoothing': False,  # label smoothing (real=0.9, fake=0.1)
                    'da_adv_both': False, # if True use both fake and real in advloss
                    
                    'da_disc_width': 256, # discriminator width
                    'da_disc_layer': 3, # discriminator layers'
                    'da_disc_set_optim': False, # if False, optimizations are the same as main net, True. set differently
                    
                    'da_disc_lr_init': 1e-4, # learning rate for discriminator (Adam)
                    'da_disc_adam_betas': (0.5, 0.9), # adam betas for for discriminator (Adam)
                    'da_disc_l2_weight_reg':1e-4, # L2 regulatino for discriminator
                    'da_disc_gamma':0.3, # gamma for discriminator
                    'da_disc_meta_epoch': 10, # gamma for discriminator
                    
                    'da_warmup': 0, # i_epogh<=da_warmup: seperately train cINN (w/o adv) and Discriminator
                    'da_disc_warmup_delay': 0, 
                    
                    'delay_cinn': 0 , # epoch to dealy cinn training. optimization starts from i_epoch >= delay_main (+da_warmup)
                    'delay_disc': 0, # epoch to dealy discriminator training. optimization starts from i_epoch >= delay_main((+da_warmup))
                    'da_stop_da': None, # epoch to stop DA (both disc and adv update)
                    # weight schedulers are set as the same as main cINN
                    
                    # train_noisy_obs deprecated
                    # 'train_noisy_obs': False,
                    # 'n_noise_MC': None,
                    # 'noise_fsigma': None,
                    
                    # Noise-Net: prenoise training (noise as condition)
                    'prenoise_training': False, # T/F
                    'unc_corrl': None, # 'Poisson', 'Ind_Unif', 'Ind_Man', 'Single', 'Seg_Unif',
                    'unc_sampling': None, # 'gaussian', 'uniform'
                    'n_sig_MC': None,  # deprecated
                    'n_noise_MC': None, # deprecated
                    # for Poisson
                    'lsigb_mean': None, 'lsigb_std': None, # p( log(sig_b) ) = G(lsigb_mean, lsigb_std)
                    'lsigb_min': None, 'lsigb_max':None,
                    # for general uniform or gaussian samplings in Ind_Unif, Ind_Man, Single, Seg_Unif
                    'lsig_mean': None, 'lsig_std':None, # array for Ind_Man, value for Ind_Unif
                    'lsig_min': None, 'lsig_max': None, 
                    # when using wavelength segment
                    'wl_seg_size': None, # wavelength segmentsize, in AA. 200 or 500A recommended
                    
                    # using flag (floag is additional conition)
                    'use_flag': False, # T/F
                    'flag_dic': None, # dictionay: key=name of flag, item = list of y_names to turn on/off (requried)
                    'flag_names': None, # list of keys of flag_dic (auto)
                    'flag_index_dic': None, # dictionaly; keya name of flag, item = list of indicees of corresponding y (auto) 
                    
                    # [TEST] Selection of conditioning network (feature network)
                    'cond_net_code': 'linear', # default is linear to maintain other networks
                                            # linear: use FeatureNet in model file. This is fully connected network. 
                                            #       Control with feature_width, feature_layer, y_dim_features in config
                                            # hybrid_cnn: use HybridFeatureNet in feature_net.py. Combination of CNN and global_net (linear)
                                            #       Control with two config_dictionaries: conv_net_config, global_net_config
                                            # cnn: use ConvolutionalNetwork as a feature net. give  conv_net_config
                                            # hybrid_stack: use HybridStackedFeatureNet in feature_net.py. Combination of CNN and linear global net, 
                                            # but stack ouput of each convolutional layer. multiple use of global output. 
                    'conv_net_config': {"in_dim_conv": None, "out_dim_conv":256, #"n_layers":3, 
                                        "in_channels":1, "start_channels":16,
                                        "kernel_size_filter": 3, "kernel_size_pooling":2,
                                        "stride_filter":2, "stride_pooling":2,
                                        "pooling_type":"max", # 'max' or 'avg'
                                        "stack_final_layers": None, # Only used in hybrid_stack. None (layer=1) or "Auto4" (reduce dimension 1/4.)
                                        },
                    
                    'global_net_config':  {"in_dim_global": None, "out_dim_global":4, "n_layers_global":2,}, 
                                  
                    }
                    
    __depreacted_param ={
                    'architecture': 'cINN', # DEPREACATED (not used anymore, but leave to read old configs)
                    'FrEIA_ver': 0.2, # version of FrEIA (2022.1.10: 0.1 or 0.2) (2023.08.10. DEPRECATED)
                    }

    
    # parameters to be set
    __cINN_parameter_arg = {**__data_param, **__dependent_param,  **__setting_param}
    
    # Read default hyperparameters from cINN_parameters.py in the same directory
    from . import cINN_parameters as p # 같은위치에서는 이게 안됨...?
    # import cINN_parameters as p # 이 위치에서는 이거만
    
    
    __parameter_kwarg = {}
    for v in dir(p):
        if v[0]=='_': continue
        # if v=='np': continue # ther 
        # save default values 
        __parameter_kwarg[v] = eval('p.%s'%(v))
        
    __class_parameter = {
                        'config_file': None, # full path of config file
#                         '_projdir': None, # specify abspath where outputs will be saved
                        } 
    
    # these are added later. don't want to print.
    __hidden_parameter = {
                        'mu_x': None, 'mu_y': None, 'w_x': None, 'w_y': None,   # rescale parameters 
                        'mu_s':None, 'w_s':None, # Noise-Net, sigma
                        'mu_f':None, 'w_f':None, # rescale params for flag
                        'mu_wl':None, 'w_wl':None, # wavelength coupling
                        'network_model': None,
                            }
        
    __parameter_default_dic = {**__parameter_kwarg, **__cINN_parameter_arg , **__class_parameter, **__depreacted_param}
    
    __slots__ =  list(__parameter_default_dic.keys())+ list(__hidden_parameter.keys()) + ['_projdir']  
    
    
    def __init__(self, **kwarg):    
        self.reset()
        self._projdir = None
        for key, value in kwarg.items():
            if key == 'proj_dir': # if you add this as keyword
                self._projdir = value
            else:
                try:
                    setattr(self, key, value)
                except Exception as e:
                    print(e)
                
        self.update_dimension()
        if self._projdir is not None:
            self.update_proj_dir()
        self.update_da_optim()

    
    # when you first declare, you will have this default setting
    def reset(self):
        for param, default_value in cINNConfig.__parameter_default_dic.items():
            setattr(self, param, default_value)
        for param, default_value in cINNConfig.__hidden_parameter.items():
            setattr(self, param, default_value)
        
            
            
    def update_dimension(self):
        if self.x_names is not None:
            self.x_dim = len(self.x_names)
        if self.y_names is not None:
            self.y_dim_in = len(self.y_names)
            if self.prenoise_training==True:
                self.y_dim_in += len(self.y_names)
            if self.use_flag==True:
                # add auto attr and update dimension
                self.flag_names = list(self.flag_dic.keys())
                self.flag_index_dic = {}
                for key, names in self.flag_dic.items():
                    self.flag_index_dic[key] = [self.y_names.index(name) for name in names]
                self.y_dim_in += len(self.flag_names)
            if self.wavelength_coupling==True:
                self.y_dim_in += len(self.y_names)
                
        if self.cond_net_code=="hybrid_cnn" or self.cond_net_code=="hybrid_stack":
            # give in_dim to conv_net and global_net -> count global params and spectral points
            self.conv_net_config['in_dim_conv'] = sum(1 for s in self.y_names if s.startswith('l') and s[1:].isdigit())
            if self.conv_net_config['in_dim_conv']==0: # flux-net
                self.conv_net_config['in_dim_conv'] = sum(1 for s in self.y_names if s.startswith('f') and is_float(s[1:]))
            self.global_net_config['in_dim_global'] = len(self.y_names) - self.conv_net_config['in_dim_conv']
            if self.prenoise_training==True:
                self.conv_net_config["in_channels"]=2
                self.global_net_config['in_dim_global'] *= 2
            else: 
                self.conv_net_config["in_channels"]=1
            
            # hybrid_stack does not use out_dim_conv in conv_net_config
            if self.cond_net_code=="hybrid_stack":
                self.conv_net_config.pop("out_dim_conv", None) # remove this key

            
    def update_proj_dir(self, change_filename=True, change_tablename=True): #, change_expander=True): # expander deprecated
        projdir = self._projdir
        if projdir[-1]!='/':
            projdir = projdir + '/'
        if change_filename:
            self.filename = projdir + self.filename
        if change_tablename:
            self.tablename = projdir + self.tablename
            if self.domain_adaptation:
                self.real_database = projdir + self.real_database
            if self.slab_grid is not None:
                self.slab_grid = projdir + self.slab_grid
        # if change_expander:
        #     self.expander = projdir + self.expander
        
    def set_proj_dir(self, projdir, **kwarg):
        self._projdir = projdir
        self.update_proj_dir(**kwarg)        
    
    def get_proj_dir(self):
        return self._projdir
    proj_dir = property(get_proj_dir) 
    
    def update_da_optim(self):
        if self.da_disc_set_optim != True:
            optim_params = ['lr_init','adam_betas', 'l2_weight_reg', 'gamma','meta_epoch']
            for param in optim_params:
                setattr(self, 'da_disc_'+param, getattr(self, param))
    
    
    # Rescale parameters
    def load_rescale_params(self):   
        state_dicts = torch.load(self.filename, map_location=torch.device('cpu'), weights_only=False)
        self.mu_x = state_dicts['rescale_params']['mu_x']#.astype(np.float32)
        self.mu_y = state_dicts['rescale_params']['mu_y']#.astype(np.float32)
        self.w_x = state_dicts['rescale_params']['w_x']#.astype(np.float32)
        self.w_y = state_dicts['rescale_params']['w_y']#.astype(np.float32)
        
        if self.prenoise_training:
            self.mu_s = state_dicts['rescale_params']['mu_s']#.astype(np.float32)
            self.w_s = state_dicts['rescale_params']['w_s']#.astype(np.float32)
        
        if self.use_flag:
            self.mu_f = state_dicts['rescale_params']['mu_f']#.astype(np.float32)
            self.w_f = state_dicts['rescale_params']['w_f']#.astype(np.float32)

        if self.wavelength_coupling:
            self.mu_wl = state_dicts['rescale_params']['mu_wl']#.astype(np.float32)
            self.w_wl = state_dicts['rescale_params']['w_wl']#.astype(np.float32)
      
    
    def obs_to_y(self, observations):
        # y = np.log(observations)
        y = np.dot(observations - self.mu_y, self.w_y)
        return y # np.clip(y, -5, 5)
    
    def y_to_obs(self, y):
        obs = np.dot(y, np.linalg.inv(self.w_y)) + self.mu_y
        return obs #np.exp(obs)
 
    def params_to_x(self, parameters):
        return np.dot(parameters - self.mu_x, self.w_x)
    
    def x_to_params(self, x):
        return np.dot(x, np.linalg.inv(self.w_x)) + self.mu_x
    
    def unc_to_sig(self, uncertanties):
        sig = np.log10(uncertanties)
        return np.dot(sig - self.mu_s, self.w_s)
        
    def sig_to_unc(self, sig):
        unc = np.dot(sig, np.linalg.inv(self.w_s)) + self.mu_s
        return 10**unc
    
    def flag_to_rf(self, flags):
        return np.dot(flags - self.mu_f, self.w_f)
    
    def rf_to_flag(self, rfs):
        return np.dot(rfs, np.linalg.inv(self.w_f)) + self.mu_f
    

    def wl_to_lambda(self, wavelength):
        return np.dot(wavelength - self.mu_wl, self.w_wl)
    
    def lambda_to_wl(self, lambdas):
        return np.dot(lambdas, np.linalg.inv(self.w_wl)) + self.mu_wl
    
    """
    
    def obs_to_y(self, observations):
        if type(observations)==torch.Tensor:
            y = torch.log(observations)
            y = torch.matmul(y - self.mu_y, torch.Tensor(self.w_y))
            return torch.clip(y, -5, 5)
        else:
            y = np.log(observations)
            y = np.dot(y - self.mu_y, self.w_y)
            return np.clip(y, -5, 5)
    
    def y_to_obs(self, y):
        if type(y)==torch.Tensor:
            obs = torch.matmul(y, torch.linalg.inv(torch.Tensor(self.w_y))) + self.mu_y
            return torch.exp(obs)
        else:
            obs = np.dot(y, np.linalg.inv(self.w_y)) + self.mu_y
            return np.exp(obs)
    
    def params_to_x(self, parameters):
        if type(parameters)==torch.Tensor:
            return torch.matmul(parameters - self.mu_x, self.w_x)
        else:
            return np.dot(parameters - self.mu_x, self.w_x)
    
    def x_to_params(self, x):
        if type(x)==torch.Tensor:
            return torch.matmul(x, torch.linalg.inv(self.w_x)) + self.mu_x
        else:
            return np.dot(x, np.linalg.inv(self.w_x)) + self.mu_x
        
    def unc_to_sig(self, uncertanties):
        if type(uncertanties)==torch.Tensor:
            sig = torch.log10(uncertanties)
            return torch.matmul(sig - self.mu_s, self.w_s)
        else:
            sig = np.log10(uncertanties)
            return np.dot(sig - self.mu_s, self.w_s)
        
    def sig_to_unc(self, sig):
        if type(sig)==torch.Tensor:
            unc = torch.matmul(sig, torch.linalg.inv(self.w_s)) + self.mu_s
            return 10**unc
        else:
            unc = np.dot(sig, np.linalg.inv(self.w_s)) + self.mu_s
            return 10**unc
     """   
    
    
    def load_network_model(self):
        self.network_model = eval(self.model_code)(self)
        self.network_model.load(self.filename, device=self.device)
        self.network_model.eval() # start with evaluation mode
    
    
    
    # Get default values
    @property
    def parameter_default_dic(self):
        return cINNConfig.__parameter_default_dic
    
    # def get_parameter_default_dic(self):
    #     return cINNConfig.__parameter_default_dic
    # parameter_default_dic = property(get_parameter_default_dic)
    
    @property
    def parameter_list(self):
        return list(cINNConfig.__parameter_default_dic.keys())
    
    @property
    def hidden_parameter_list(self):
        return list(cINNConfig.__hidden_parameter.keys())
    
#    @property
#    def relevant_parameter_list(self):
#        _params = list(cINNConfig.__parameter_default_dic.keys())
#        # flags
#        if self.checkpoint_save != True:
#            for param in ["checkpoint_save_interval","checkpoint_save_overwrite"]:
#                if param in _params: _params.remove(param)
#        if self.model_code == "ModelAdamGLOW":
    
    @property
    def printable_parameter_list(self):
        _ = list(cINNConfig.__parameter_default_dic.keys())
        if len(self.x_names) > 20:
            _.remove(self.x_names)
        if len(self.y_names) > 20:
            _.remove(self.y_names)
        return _
    
    # show current config setting
    @property
    def config_setting(self):
        dic = {}
        for param in self.parameter_list:
            # if param not in cINNConfig.__unreadable_parameter:
            dic[param] = getattr(self, param)
        return dic
    
    # Show name and values of must set attributes
    @property
    def must_set_parameters(self):
        dic = cINNConfig.__cINN_parameter_arg.copy()
        for param in dic.keys():
            dic[param] = getattr(self, param)
        return dic
    
    
    # parameters depending on Models
    __adam_glow = ['exponent_clamping', 'use_permutation']
    __adam_allinone = ['affine_clamping', 'gin_block', 'global_affine_init',
                       'global_affine_type', 'permute_soft','learned_householder_permutation',
                       'reverse_permutation']
    
    # Show name and values of model specific parameters (usually you don't need to control these)
    @property
    def model_parameters(self):
        dic = {}
        if self.model_code == 'ModelAdamGLOW':
            for param in cINNConfig.__adam_glow:
                dic[param] = getattr(self, param)
        elif self.model_code == 'ModelAdamAllInOne':
            for param in cINNConfig.__adam_allinone:
                dic[param] = getattr(self, param)
        return dic
            
    
    def print_short_setting(self):
        print("==================== cINN NETWORK SETTING =================")
        if self.config_file is not None:
            print("cINN_config file:", os.path.basename(self.config_file))
        print("database:", os.path.basename(self.tablename))
        # print("DB expander:", os.path.basename(self.expander))
        print("# of parameters:", self.x_dim)
        print("# of observables:", self.y_dim_in)
        print("device:",self.device)
        print("===========================================================") 
        
    def import_expander(self): # expander가 cINN_set으로 들어가면, 그리고 고정으로 결정되면 이 부분도 간단하게 바뀔 것. 
        # 사실상 하나의 프로젝트에서 expander가 여러개 필요할지는 모르겠다. 아마 하나면 될 것 같은데 가능성을 둬야할까?
        # expander_path = self.expander
        # if os.path.exists(expander_path):
        #     if expander_path[-3:]!='.py':
        #         expander_path+='.py'       
        #     try:
        #         spec = importlib.util.spec_from_file_location( os.path.basename(expander_path).split('.')[0], expander_path)
        #         db_exp = importlib.util.module_from_spec(spec)
        #         spec.loader.exec_module(db_exp)
        #         # read using a function in expander module
        #     except Exception as e:
        #         sys.exit(e)
        #     return db_exp
        # else:
        #     sys.exit('There is no expander (%s)'%expander_path)
            
        try: 
            from . import expander as db_exp
            # import expander as db_exp
        except Exception as e:
            sys.exit(e)
        return db_exp
        
    def copy(self, keep_network=False):
        new_config = self.__class__.__new__(self.__class__)
        
        all_slots = set()
        for cls in self.__class__.__mro__:
            slots = getattr(cls, '__slots__', [])
            if isinstance(slots, str):
                all_slots.add(slots)
            else:
                all_slots.update(slots)
        for key in all_slots:
            if hasattr(self, key):
                if key == 'network_model' and not keep_network:
                    # 메모리 점유의 주범은 그냥 None 처리
                    setattr(new_config, key, None)
                else:
                    value = getattr(self, key)
                    setattr(new_config, key, copy.deepcopy(value))
                    
        return new_config
        # return copy.deepcopy(self)
    
    #################################################################
    
    def read_config(self, config_file, proj_dir=None, verbose=True,
                    change_filename=True, change_tablename=True): #, change_expander=True):
        # declare empty class and read python file
        # import importlib.util
        if config_file[-3:]!='.py':
            config_file+='.py'
            
        try:
            spec = importlib.util.spec_from_file_location( os.path.basename(config_file).split('.')[0], config_file)
            _c = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_c)
            
            # find relevant parameters and assign attribute
            for v in dir(_c):
                if v in self.parameter_list:
                    try:
                        setattr(self, v, eval('_c.%s'%(v)))
                    except Exception as e:
                        print(e)
            self.update_dimension()
            self.update_da_optim()
            self.config_file = config_file
            if proj_dir is not None:
                self.set_proj_dir(proj_dir, change_filename=change_filename, 
                                  change_tablename=change_tablename)#, change_expander=change_expander)
            
            try:
                self.load_rescale_params()
            except Exception as e:
                if verbose:
                    print('(Skip for first training) Cannot load rescale parameters: %s'%e)

        except Exception as e:
            sys.exit(e)
            
    def parse_from_file(self, file):
        text = open(file, 'r').read()
        # Remove comments to avoid confusions
        text = re.sub(r'("""|\'\'\')(.*?)\1', '', text, flags=re.DOTALL)
        
        lines = text.split('\n')
        # remove any line starts with #
        config_lines = []
        for line in lines:
            stripped = line.split('#')[0].rstrip()  # # 이후 삭제
            if stripped:  # 빈 줄은 제외
                config_lines.append(stripped)
        
        # find line with = : start line of varaiable definition
        i_def_line = np.where( np.array([("=" in comp) and (len(comp.split('='))==2) for comp in config_lines]))[0]
        
        parse_dic = {}
        for j, ind in enumerate(i_def_line):
            # ind is the start of definition. Find varaible name 
            name = config_lines[ind].split('=')[0].replace(' ', '')
            # collect all data from ind to before i_def_line[j+1]
            if j==len(i_def_line)-1:
                data_lines = config_lines[ind:]
            else:
                data_lines = config_lines[ind: i_def_line[j+1]]
            data_list = [ config_lines[ind].split('=')[1] ]
            if len(data_lines)>1:
                for dd in data_lines[1:]:
                    data_list.append(dd)
            data_str = '\n'.join(data_list)
            
            parse_dic[name] = data_str
        return parse_dic
            
    # def find_str_names(self, config_file, dim_max=20):
    
    #     n_x = len(self.x_names); n_y = len(self.y_names)
    #     if n_x <= dim_max and n_y <= dim_max:
    #         return None, None
    #     else:
    #         with open(config_file, 'r') as f:
    #             text = f.read()
        
    #         # Remove comments to avoid confusions
    #         text = re.sub(r'("""|\'\'\')(.*?)\1', '', text, flags=re.DOTALL)
        
    #         lines = text.split('\n')
    #         config_components = []
    #         for line in lines:
    #             stripped = line.split('#')[0].rstrip()  # # 이후 삭제
    #             if stripped:  # 빈 줄은 제외
    #                 config_components.append(stripped)
            
    #         if n_x > dim_max:
    #             i_comp = np.where( np.array([("x_names" in comp) and ("#" not in comp) for comp in config_components]))[0][0]
    #             str_x_names = config_components[i_comp].split("=")[-1].strip()
    #         else:
    #             str_x_names = None
                
    #         if n_y > dim_max:
    #             i_comp = np.where( np.array([("y_names" in comp) and ("#" not in comp) for comp in config_components]))[0][0]
    #             str_y_names = config_components[i_comp].split("=")[-1].strip()
    #         else:
    #             str_y_names = None
                
    #         return str_x_names, str_y_names
                
    
    
    def write_config(self, config_comment = None, str_x_names = None, str_y_names=None,
                     use_str_names=False,
                     str_flag_items=None):
        # usually write short config_name for config_comment
        
        def new_line(input_list): # to make 1 blank lines
            input_list.append(' ')
        
        def simple_sentence(config, key, dtype=None, comment = None):
            param = getattr(config, key)
            if dtype is not None:
                param = dtype(param)
            if isinstance(param, str):
                txt = "%s = '%s'"%(key, param) # Else the str will miss the quotation marks
            else:
                txt = "%s = %s" %(key, param)    
            if comment is not None:
                txt += " # %s"%comment
            return txt
        
        def format_list(name, items, max_items_per_line=10):
            if len(items) <= max_items_per_line:
                return f"{name} = [\n\t'" + "',\n\t'".join(items) + "'\n]"
            else:
                lines = []
                for i in range(0, len(items), max_items_per_line):
                    chunk = items[i:i + max_items_per_line]
                    line = ", ".join(f"'{item}'" for item in chunk)
                    lines.append(line)
                return f"{name} = [\n\t" + ",\n\t".join(lines) + "\n]"
        
        # write cINN_parameter_arg first (something to control)
        
        # START
        contents = []
        
        if config_comment is not None:
            contents.append('"""%s"""'%config_comment)
        else:
            contents.append('""" cINN configuration file """')
        new_line(contents)
        
        
        # x_names, y_names, tablename, expander
        contents.append( "# x_names and y_names according to DB (tablename) column names")
       
        # get string from config_file
        if use_str_names==True:
            # str_x_names, str_y_names = self.find_str_names(self.config_file)
            parse_dic = self.parse_from_file(self.config_file)
            str_x_names = parse_dic.get('x_names', None)
            str_y_names = parse_dic.get('y_names', None)
            
        if str_x_names == None:
            # contents.append( "x_names = [\n\t'%s'\n]"%("' ,\n\t'".join(self.x_names))  )
            contents.append(format_list("x_names", self.x_names, max_items_per_line=5))
        else:
            contents.append( "x_names = %s"%(str_x_names) )
        new_line(contents)
    
        if str_y_names == None:
            # contents.append( "y_names = [\n\t'%s'\n]"%("' ,\n\t'".join(self.y_names))  )
            contents.append(format_list("y_names", self.y_names, max_items_per_line=10))
        else:
            contents.append( "y_names = %s"%(str_y_names) )
        new_line(contents)
        
        if self.random_parameters is not None:
            contents.append( "# These parameters will be randomly selected during the training from corresponding ranges")
            contents.append(simple_sentence(self, "random_parameters"))
        new_line(contents)
        
        if self.additional_kwarg is not None:
            contents.append( "# These are kwarg needed in some functions like extinction ")
            contents.append(simple_sentence(self, "additional_kwarg"))
        new_line(contents)
        # comment on using proj_dir?
        
        # filename: name code of all outputs
        contents.append("# name of output network, Ex) output/xxxx_network.pt")
        contents.append("# Do not forget to make upper directory if you use")
        contents.append( "filename = '%s'"%(self.filename)   )
        new_line(contents)
        
        # tablename, expander
        # contents.append("# name (inculuding path) of database and corresponding expander")
        contents.append("# name (inculuding path) of database")
        for param in ["tablename"]:#, "expander"]:
            contents.append(simple_sentence(self, param))
        new_line(contents)
        
        contents.append( "# if you use network outside your proj_dir")
        contents.append( "# you can set proj_dir by > read_config_from_file('path_to_configfile',  proj_dir = proj_dir)" )
        contents.append( "# This automatically updates filename and tablename: proj_dir+filename" )
        contents.append( "# you can control with keywords:") 
        contents.append( "# change_filename=True/False, change_tablename=True/False" )
        new_line(contents)
        
        # model_code
        contents.append( "# Set model_code:'ModelAdamGLOW' , 'ModelAdamAllInOne'" )
        # contents.append( "sEIA_ver = %s"%(self.FrEIA_ver) )
#        for param in ["architecture", "model_code", "FrEIA_ver"]:
        for param in ["model_code"]:
            contents.append(simple_sentence(self, param))
        new_line(contents)
        
        # TEST
        contents.append( "# Set cond_net_code:'linear' , 'hybrid_cnn', 'hybrid_stack'. if not set, default is linear" )
        contents.append(simple_sentence(self, 'cond_net_code'))
        if  self.cond_net_code=="hybrid_cnn" or self.cond_net_code=="hybrid_stack":
            contents.append(simple_sentence(self, "conv_net_config"))
            contents.append(simple_sentence(self, "global_net_config"))
        new_line(contents)
        
        # test_frac
        contents.append( "# fraction of test set: 0 < test_frac < 1")
        contents.append(simple_sentence(self, "test_frac"))
        new_line(contents)
        
        # smoothing parameters
        contents.append( "# smoothing discretized parameters (adding Gaussian noise)")
        contents.append(simple_sentence(self, "train_smoothing"))
        if self.train_smoothing:
            contents.append(simple_sentence(self, "smoothing_sigma"))
        new_line(contents)
        
        # Additional Normalization processes for spectra
        contents.append( "# Normalize spectra (option: normalize_total_flux, normalize_mean_flux)")
        contents.append(simple_sentence(self, "normalize_flux"))
        if self.normalize_flux:
            contents.append(simple_sentence(self, "normalize_total_flux"))
            contents.append(simple_sentence(self, "normalize_mean_flux"))
            contents.append(simple_sentence(self, "normalize_f750"))
        new_line(contents)
        
        # Option on veiling
        if self.use_Hslab_veiling:
            contents.append( "# Veiling option (using Hydrogen Slab model for veiling)")
            contents.append(simple_sentence(self, "use_Hslab_veiling"))
            contents.append(simple_sentence(self, "use_one_Hslab_model", comment="If True, it use one example slab model for all. slab_grid is ignored"))
            contents.append(simple_sentence(self, "slab_grid", comment="Path to slab grid file (.csv)"))
            contents.append("# If use_Hslab_veiling=True but use_one_Hslab_model !=True, then automatically read slab_grid")
        new_line(contents)
        
        # [deprecated] if you turn on noisy training
        # contents.append(simple_sentence(self, "train_noisy_obs"))
        # contents.append( "train_noisy_obs = %s"%(self.train_noisy_obs) )
        # if self.train_noisy_obs == True:
        #     contents.append( "n_noise_MC = %s"%(int(self.n_noise_MC)) )
        #     contents.append( "noise_fsigma = %s"%(self.noise_fsigma) )
        # new_line(contents)  
        
        # if your use prenoise training (N/S as additional condition)
        contents.append('# Prenoise training (Noise-Net): Use dY/Y as additional conditions')
        contents.append(simple_sentence(self, "prenoise_training"))
        if self.prenoise_training == True:
            # contents.append(simple_sentence(self, "n_sig_MC", comment="This must be 1 (will be deprecated)"))
            # contents.append(simple_sentence(self, "n_noise_MC", comment="This must be 1 (will be deprecated)"))
            
            contents.append('# Correlation between uncertainies in one obs')
            contents.append(simple_sentence(self, "unc_corrl"))
            contents.append('# Distribution of uncertainty sampling')
            contents.append(simple_sentence(self, "unc_sampling"))
            _to_write = []
            # if self.unc_corrl == 'Ind_Man' or self.unc_corrl == 'Ind_Unif':
            if self.unc_sampling == 'gaussian':
                _to_write += ["lsig_mean", "lsig_std"]
            elif self.unc_sampling == 'uniform':
                _to_write += ["lsig_min", "lsig_max"]
            if 'Seg' in self.unc_corrl:
                _to_write += ['wl_seg_size']
            for param in _to_write:
                contents.append(simple_sentence(self, param))
        new_line(contents)
        
        # if you use flag in Normal-Net
        contents.append('# Network with y flag : control turning on/off of certain y components')
        contents.append(simple_sentence(self, "use_flag") )
        if self.use_flag == True:
            contents.append('# Dictionary of flags to make: key=flag_name, item=list of y_names')
            if  str_flag_items is not None:
                txt = ["{"]
                for i, key in enumerate(self.flag_names):
                    txt.append("'%s': %s, "%(key, str_flag_items[i]))
                txt.append("}")
                contents.append( 'flag_dic = '+'\n\t'.join(txt))
            else:
                contents.append(simple_sentence(self, "flag_dic"))
        new_line(contents)

        # if you use wavelength coupling
        contents.append('# Network with wavelength coupling')
        contents.append(simple_sentence(self, "wavelength_coupling") )
        new_line(contents)
            
        # If you use Domain Adaptaion 
        contents.append('# Domain Adaptaion: Use real data to improve simulation gap')
        contents.append(simple_sentence(self, "domain_adaptation"))
        if self.domain_adaptation == True:
            contents.append("# Path to the real database to be used for domain adaptation")
            contents.append(simple_sentence(self, "real_database", comment="path will be updated with proj_dir"))
            contents.append(simple_sentence(self, "real_frac", comment="fraction of real data with repect to the batch size (real_frac x batchsize)"))
            contents.append(simple_sentence(self, "da_mode", comment="Mode to calculate discriminator loss :simple/WGAN"))
            if self.da_mode=='simple':
                contents.append(simple_sentence(self, "lambda_adv", comment="Loss = NLL + lambda_adv * L_adv, >0"))
            contents.append(simple_sentence(self, "da_disc_train_step", comment="how frequently update weight: 1=every batch, if None, it automatically increases from 1 dependin on epoch"))
            contents.append(simple_sentence(self, "delay_cinn", comment="epoch to dealy cinn training. optimization starts from i_epoch >= delay_main"))
            contents.append(simple_sentence(self, "delay_disc", comment="epoch to dealy discriminator training. optimization starts from i_epoch >= delay_disc"))
            
            
            contents.append("# Discriminator construction")
            contents.append(simple_sentence(self, "da_disc_width", comment="Width of Discriminator"))
            contents.append(simple_sentence(self, "da_disc_layer", comment="# of layers of Discriminator"))
            contents.append(simple_sentence(self, "da_disc_set_optim", comment="if False, optimization parameters are the same as main cINN, True. set differently"))
            if self.da_disc_set_optim:
                contents.append("# Discriminator learning optimization")
                for param in ["gamma", "lr_init", "l2_weight_reg", "adam_betas", "meta_epoch"]:
                    contents.append(simple_sentence(self,'da_disc_'+param))
            # contents.append('# Currently, discriminator optimization uses adam betas, lr_init, meta_epoch, gamma from main network')
        new_line(contents)
        
        
        # ============ parameters in cNN_parameters.py ===========
        # training hyperparameters: device, batchsize, epoch, 
        contents.append('"""\n Training hyperparameters \n"""' )
        contents.append(simple_sentence(self, "device"))
        contents.append( "# device: cpu, cuda, cuda:0 (modifiable anytime after you read config)")
        
        for param in ["batch_size", "n_epochs"]:
            contents.append(simple_sentence(self, param, dtype=int))
        new_line(contents)
        for param in ["pre_low_lr"]:
            contents.append(simple_sentence(self, param))
        if np.log2(self.n_its_per_epoch).is_integer():
            contents.append( "n_its_per_epoch = 2**%s"%(int(np.log2(self.n_its_per_epoch)))  )
        else:
            contents.append( "n_its_per_epoch = %s"%(self.n_its_per_epoch) )
        new_line(contents)
        
        contents.append(simple_sentence(self, "load_file"))
        new_line(contents)
        
        for param in ["checkpoint_save", "checkpoint_save_interval", "checkpoint_save_overwrite", "checkpoint_remove_after_training"]:
            contents.append(simple_sentence(self, param))
        new_line(contents)
        
        
        # Learning Optimization
        contents.append('"""\n Learning Optimization \n"""' )
        for param in ["gamma", "lr_init", "l2_weight_reg", "adam_betas"]:
            contents.append(simple_sentence(self, param))
        contents.append(simple_sentence(self, "meta_epoch", dtype=int))
        contents.append( "# how often you change the learning rate (1=every epch)" )
        new_line(contents)
        contents.append(simple_sentence(self, "seed_weight_init", dtype=int))
        contents.append( "# Seed for the random network weight initialisation. The default is 1234." )
        new_line(contents)
        contents.append(simple_sentence(self, "do_rev"))
        if self.do_rev:
            contents.append(simple_sentence(self, "latent_noise"))
        new_line(contents)
        
        # Model construnction
        contents.append('"""\n Model Construction \n"""' )
        contents.append(simple_sentence(self, "n_blocks", dtype=int))
        new_line(contents)
        contents.append( "# subnetwork's internal width, the number of layers")
        for param in ["internal_width", "internal_layer"]:
            contents.append(simple_sentence(self, param, dtype=int)) 
        contents.append(simple_sentence(self, "init_scale"))
        contents.append( "# feature_net (cond_net): (y is converted to feature through cond_net to be used as conditions)")
        contents.append(simple_sentence(self, "y_dim_features", dtype=int)) 
        contents.append( "# dimension used in cond_net ")
        for param in ["feature_width", "feature_layer"]:
            contents.append(simple_sentence(self, param, dtype=int)) 
        contents.append( "# cond_net's width and layers")
        new_line(contents)
        
        if self.model_code == 'ModelAdamGLOW':
            _to_write = ["exponent_clamping", "use_permutation"]
            contents.append("# ModelAdamGLOW hyperparameters (%d)"%len(_to_write))
        elif self.model_code == 'ModelAdamAllInOne':
            _to_write = ["affine_clamping", "gin_block",
                        "global_affine_init", "global_affine_type",
                        "permute_soft", "learned_householder_permutation",
                        "reverse_permutation"]
            contents.append("# ModelAdamAllInOne hyperparameters (%d)"%len(_to_write))
        else:
            _to_write = []
        for param in _to_write:
            contents.append(simple_sentence(self, param))
        new_line(contents)    
        
        
        # Visualization
        contents.append('"""\n Visualization \n"""' )
        for param in ["loss_names", 
                      "loss_plot_xrange", "loss_plot_yrange",
                     "progress_bar", "live_visualization"]:
            contents.append(simple_sentence(self, param))
        new_line(contents)
    
        return contents
    
    
    
    def print_config(self, **kwarg):
        contents = self.write_config(**kwarg)
        print('--------------------------------------\n')
        print('\n'.join(contents))
        print('--------------------------------------\n')
        
        
    def save_config(self, config_file=None, verbose=True, **kwarg):
        # you can either set your name config_file as keyword or use config.config_file
        # If you have both config.filename and filename (arg), argument filename precede config.filename
        # because config.filename could be set when you read configfile. And you may want to save this with different filename
        # config.filename is not set in this function even though config.filename was None   
        
        if config_file is None:
            if self.config_file is None:
                print('\tBoth config_file keyword and config_file attribute are None. Cannot save config file')
                return 
            else:
                config_file = self.config_file
        
        file = open(config_file, 'w')
        file.write('\n'.join(self.write_config(**kwarg)))
        file.close()
        if verbose:
            print('\tSaved config file: %s'%config_file)   
        
def read_config_from_file(config_file, **kwarg):
    
    config = cINNConfig()
    config.read_config(config_file, **kwarg)
    
    return config


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
