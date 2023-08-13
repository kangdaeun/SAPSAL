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
                    
                    # Randomizing parameter on the fly
                    'random_parameters': None, # dictionary
                    'additional_kwarg': None, # dictionary : currently used for: Rv
                    
                    # train_noisy_obs deprecated
                    # 'train_noisy_obs': False,
                    # 'n_noise_MC': None,
                    # 'noise_fsigma': None,
                    
                    # prenoise training (noise as condition)
                    'prenoise_training': False, # T/F
                    'unc_corrl': None, # 'Poisson', 'Ind_Unif', 'Ind_Man', 'Single'
                    'unc_sampling': None, # 'gaussian', 'uniform'
                    'n_sig_MC': None, 
                    'n_noise_MC': None,
                    # for Poisson
                    'lsigb_mean': None, 'lsigb_std': None, # p( log(sig_b) ) = G(lsigb_mean, lsigb_std)
                    'lsigb_min': None, 'lsigb_max':None,
                    # for Ind_Unif and Ind_Man
                    'lsig_mean': None, 'lsig_std':None, # array for Ind_Man, value for Ind_Unif
                    'lsig_min': None, 'lsig_max': None, 
                    
                    # using flag (floag is additional conition)
                    'use_flag': False, # T/F
                    'flag_dic': None, # dictionay: key=name of flag, item = list of y_names to turn on/off (requried)
                    'flag_names': None, # list of keys of flag_dic (auto)
                    'flag_index_dic': None, # dictionaly; keya name of flag, item = list of indicees of corresponding y (auto) 
                                  
                    }
                    
    __depreacted_param ={
                    'architecture': 'cINN', # DEPREACATED (not used anymore, but leave to read old configs)
                    'FrEIA_ver': 0.2, # version of FrEIA (2022.1.10: 0.1 or 0.2) (2023.08.10. DEPRECATED)
                    }
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
                        'mu_s':None, 'w_s':None, 
                        'mu_f':None, 'w_f':None, # rescale params for flag
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
                self.y_dim_in *= 2
            if self.use_flag==True:
                # add auto attr and update dimension
                self.flag_names = list(self.flag_dic.keys())
                self.flag_index_dic = {}
                for key, names in self.flag_dic.items():
                    self.flag_index_dic[key] = [self.y_names.index(name) for name in names]
                self.y_dim_in += len(self.flag_names)
            
    def update_proj_dir(self, change_filename=True, change_tablename=True): #, change_expander=True): # expander deprecated
        projdir = self._projdir
        if projdir[-1]!='/':
            projdir = projdir + '/'
        if change_filename:
            self.filename = projdir + self.filename
        if change_tablename:
            self.tablename = projdir + self.tablename
        # if change_expander:
        #     self.expander = projdir + self.expander
        
    def set_proj_dir(self, projdir, **kwarg):
        self._projdir = projdir
        self.update_proj_dir(**kwarg)        
    
    def get_proj_dir(self):
        return self._projdir
    proj_dir = property(get_proj_dir) 
    
    
    # Rescale parameters
    def load_rescale_params(self):   
        state_dicts = torch.load(self.filename, map_location=torch.device('cpu'))
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
        
    def copy(self):
        return copy.deepcopy(self)
    
    #################################################################
    
    def read_config(self, config_file, proj_dir=None, verbose=True,
                    change_filename=True, change_tablename=True): #, change_expander=True):
        # declare empty class and read python file
        import importlib.util
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
            self.config_file = config_file
            if proj_dir is not None:
                self.set_proj_dir(proj_dir, change_filename=change_filename, 
                                  change_tablename=change_tablename)#, change_expander=change_expander)
            
            try:
                self.load_rescale_params()
            except Exception as e:
                if verbose:
                    print('Cannot load rescale parameters: %s'%e)

        except Exception as e:
            sys.exit(e)
            
            
    
    
    def write_config(self, config_comment = None, str_x_names = None, str_y_names=None,
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
        # check the dim of parameters
        if  (len(self.x_names) > 30) * (str_x_names == None) :
            raise Exception("Too many items in x_names to write! \n Use str_x_names=")
        elif (len(self.y_names) > 30) * (str_y_names == None) :
            raise Exception("Too many items in y_names to write! \n Use str_y_names=")
            
        if str_x_names == None:
            contents.append( "x_names = [\n\t'%s'\n]"%("' ,\n\t'".join(self.x_names))  )
        else:
            contents.append( "x_names = %s"%(str_x_names) )
    
        if str_y_names == None:
            contents.append( "y_names = [\n\t'%s'\n]"%("' ,\n\t'".join(self.y_names))  )
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
        
        # [deprecated] if you turn on noisy training
        # contents.append(simple_sentence(self, "train_noisy_obs"))
        # contents.append( "train_noisy_obs = %s"%(self.train_noisy_obs) )
        # if self.train_noisy_obs == True:
        #     contents.append( "n_noise_MC = %s"%(int(self.n_noise_MC)) )
        #     contents.append( "noise_fsigma = %s"%(self.noise_fsigma) )
        # new_line(contents)  
        
        # if your use prenoise training (N/S as additional condition)
        contents.append('# Prenoise training (Soft training): Use dY/Y as additional conditions')
        contents.append(simple_sentence(self, "prenoise_training"))
        if self.prenoise_training == True:
            contents.append(simple_sentence(self, "n_sig_MC", comment="This must be 1 (will be deprecated)"))
            contents.append(simple_sentence(self, "n_noise_MC", comment="This must be 1 (will be deprecated)"))
            
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
        
        for param in ["checkpoint_save", "checkpoint_save_interval", "checkpoint_save_overwrite"]:
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
