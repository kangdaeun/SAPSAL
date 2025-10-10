#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 00:15:43 2021

@author: daeun

config 의 정보를 바탕으로 필요한 모듈을 사용하여 train.py (get_loader)에 사용할수있도록 loader 형성

기존 data.py에 있던 
- get_loader
- load_data
- param to x / obs _to y
- mu, w

주어진 db정보, X, Y정보를 받아서 필요한 데이터를 받아오는 역할 + 훈련에 맞게 프리프로세싱, 편집하는 역할
- 추출
- 스무딩
- 선형변환
- 분리
- 로더

DataLoader 에서 추가된 속성
- mu, w 4개
- database (whole db table)

"""
# from astropy.io import ascii
# import pandas as pd
import numpy as np
import torch # get loader
import torch.utils.data
import sys #, os
import multiprocessing
# import importlib.util

from .cINN_config import cINNConfig




def smooth(data, sigma):
    return data + sigma * np.random.randn(len(data))#.astype(np.float32)


def smooth_table(table, smoothing_sigma):
    for i, name in enumerate(table.colnames):
        if name in smoothing_sigma.keys():
            table[name] = smooth(table[name].data, smoothing_sigma[name])
            
    return table

def smooth_array(array, names, smoothing_sigma):
    names = list(names)
    for key, sigma in smoothing_sigma.items():
        if key in names:
            i = names.index(key)
            array[:,i] = smooth(array[:,i], sigma)
            
    return array
            
    

def whitening_matrix(X,fudge=1e-7):
    '''https://stackoverflow.com/questions/6574782/how-to-whiten-matrix-in-pca'''
    # get the covariance matrix
    Xcov = np.cov(X.T)
    # eigenvalue decomposition of the covariance matrix
    d, V = np.linalg.eigh(Xcov)
    # a fudge factor can be used so that eigenvectors associated with
    # small eigenvalues do not get overamplified.
    D = np.diag(1. / np.sqrt(d+fudge))
   
    W = np.dot(np.dot(V, D), V.T)
    Xw = np.dot(X, W)
    scaling = np.diag(1./np.std(Xw, axis=0))
    return np.dot(W, scaling)#.astype(np.float32)

def stddev_matrix(X):
    return np.diag(1./np.std(X, axis=0))#.astype(np.float32) # 이쪽은 불필요하긴한데


def get_num_workers():
    try:
        if torch.backends.mps.is_available():
            return 0
        if torch.cuda.is_available():
            num_cpu = multiprocessing.cpu_count()
            return min(16, max(1, num_cpu // 4))  # GPU 사용 시 최적화
        # CPU-only 환경에서는 0
        return 0
    except:
        return 0



# 속성 갯수 차단하기. config없을 때 어떻게 할지?
class DataLoader(cINNConfig):
    
    # 하위클래스로 바뀌면서 추가로 들어가야하는 값. 광역변수로 쓰던 4개 속성을 배치시키고. 이게 필요한 함수를 돌리기 전에 꼭 있어야 하는 걸    
    def __init__(self, config, update_rescale_parameters=False): # keyword: update_rescale_params
        super().__init__()
        
        # inherit attributes from config(cINNConfig variable)
        # if config is not None:
        #     self.inherit_config(config)
        self.inherit_config(config)
            
        # import and include expander 
        self.include_expander()
        
        # init 때에 전체 db를 한 번 읽어주고 그 이후로는 읽지 않는다. 최초 선언때만 읽느라 시간 걸림. 
        # 그럼 시간 걸리는 김에 계산한번 거쳐서 mu도 업뎃하자.
        self.read_database() # config를 받았던가, tablename, x_names, y_names 
        
        
        # self.mu_x = mu_x
        # self.w_x = w_x
        # self.mu_y = mu_y
        # self.w_y = w_y
        
        # update mu_x, w_x, mu_y, w_y following the training config setting
        if (self.mu_x is None) or (self.mu_y is None) or (self.w_x is None) or (self.w_y is None):
            update_rescale_parameters = True
        
        if update_rescale_parameters:
            self.update_rescale_params()   # this need .database from read_databas()
        
    # methods needed in __init__    
    def inherit_config(self, config):
        for param in config.parameter_list + config.hidden_parameter_list:
            setattr(self, param, getattr(config, param) )
      
        
        
    def read_database(self):
        tablename = self.tablename
        # x_names = self.x_names
        # y_names = self.y_names
        
        # try:
        #    whole_table = self.exp.expand_database(tablename, x_names + y_names)
        # except:
        #     db_exp = self.import_expander()
        #     whole_table = db_exp.expand_database(tablename, x_names + y_names)
        
        self.database = self.exp.read_database(tablename)
        
        
    def include_expander(self):
        self.exp = self.import_expander()

    # def get_wl(self):  # Now moved to expander
    #     # get wavelength array for specific network. default is MUSE spec (not specified in additional_kwarg['wl_in_str'])
    #     wl_in_str = self.additional_kwarg.get('wl_in_str', None)
    #     if wl_in_str is not None:
    #         return np.array(eval(wl_in_str))
    #     else:
    #         return self.exp.get_muse_wl() # np.arange(4750.1572265625, 9351.4072265625, 1.25) #from f20 (3681)
           
    
    # differce to extract_ is this can preprocess , rescale the data if you want
    def load_data(self, smoothing=False, smoothing_sigma=False, obs_clipping=False,   
                  veil_flux = False, extinct_flux = False,
                  normalize_flux=False, normalize_total_flux=False, normalize_mean_flux=False, normalize_f750 = False,
                  dummy_slab = False,
                  random_seed = 0.0,
                  **kwargs): #(kwargs = random_seed=0.0, )
        
        x_names = self.x_names
        y_names = self.y_names
        
#         data_table = self.extract_table()
        
#         # Do you want to preprocess the data?
#         # 1) smoothing
#         if smoothing: 
#             data_table = smooth_table(data_table, smoothing_sigma)
            
#         # 2) clipping for observations (not sure but used as default)
#         # assueming y > 0
#         if obs_clipping:
#             for i, name in enumerate(y_names):
#                 clip_low, clip_high = np.sort(np.abs(data_table[name][data_table[name]>0])
#                                               )[[int(0.005*len(data_table)), -int(0.005*len(data_table))]]
#                 data_table[name] = np.clip(data_table[name].data, clip_low, clip_high)
                
        # split x and y
        # all_param = np.array(data_table[x_names]).view(np.float64).reshape(-1, len(x_names))
        # all_obs = np.array(data_table[y_names]).view(np.float64).reshape(-1, len(y_names))
        
        
        if self.random_parameters is not None and self.additional_kwarg is not None:
            f_min_dic = self.additional_kwarg.get('f_min_dic', {})
            f_max_dic = self.additional_kwarg.get('f_max_dic', {})
            
        all_param, all_obs = self.exp.divide_xy(self.database, x_names, y_names,
                                                random_parameters=self.random_parameters, random_seed=random_seed,
                                                f_min_dic=f_min_dic, f_max_dic=f_max_dic,
                                                wl=self.exp.get_wl(self), # wl is needed only when using flux values (f1233.5, etc) in y_names instaed of spectra
                                                )
        # if using flux: all_obs is in tuple (all_ob, temp_obs)
        if type(all_obs)==tuple:
            all_obs, spec = all_obs
            use_spec = False
            spec_names_nested = self.exp.get_spec_names_for_flux(y_names, wl=self.exp.get_wl(self), dwl=10)
            spec_names = [item for sublist in spec_names_nested for item in sublist] # flatten
            spec_indices = self.exp.get_spec_index(spec_names)
            spec_locs = self.exp.get_flux_loc(y_names, use_bool=False)
        else:
            use_spec = True
            spec_indices = self.exp.get_spec_index(self.y_names)
            spec_locs = self.exp.get_spec_index(self.y_names, get_loc=True)
            spec = all_obs[:, spec_locs]
            spec_names = list(np.array(y_names)[spec_locs])
            
        # 1) Add veiling 
        if veil_flux:
            wl = self.exp.get_wl(self)[spec_indices]
            if 'veil_r' in x_names:
                veil_values = all_param[:, x_names.index("veil_r")]
            elif 'log_veil_r' in x_names:
                veil_values =  10**all_param[:, x_names.index("log_veil_r")]
            else:
                raise ValueError("Cannot veil: neither veil_r nor log_veil_r is in x_names.")
            
            # ordinary constant veilnig if use_Hslab_veiling is not True
            if self.use_Hslab_veiling:
                # not a grid of model but just one
                if self.use_one_Hslab_model==True: 
                    fslab_norm = self.exp.read_example_slab()[spec_indices]
                    spec = self.exp.add_slab_veil(wl, spec, veil=veil_values, fslab_750=fslab_norm)
                    # all_obs[:, spec_locs] = self.exp.add_slab_veil(wl, all_obs[:, spec_locs], veil=veil_values, fslab_750=fslab_norm)
                elif self.slab_grid is not None:
                    # This will make fslab_norm only for corresponding y_names and add slab parameters to predict based on x_names
                    # fslab_norm have N_model x N(spec_locs)
                    # this fills nan values in all_params
                    # possible slab parameters = ['Tslab', 'log_ne', 'log_tau0', 'log_Fslab']
                    all_param, fslab_norm = self.exp.assign_slab_grid(self.slab_grid, all_param, x_names, spec_names, random_seed=random_seed)
                    # add veiling effect
                    new_, veiling = self.exp.add_slab_veil(wl, spec, veil=veil_values, fslab_750=fslab_norm, return_veil=True)
                    # new_, veiling = self.exp.add_slab_veil(wl, all_obs[:,spec_locs], veil=veil_values, fslab_750=fslab_norm, return_veil=True)
                    if 'log_veil_r_6200' in x_names or 'veil_r_6200' in x_names:
                        r_6200 = self.exp.get_flux_at(wl, veiling, 6200.)/self.exp.get_flux_at(wl, all_obs[:,spec_locs], 6200.)
                        if 'log_veil_r_6200' in x_names:
                            all_param[:, x_names.index('log_veil_r_6200')] = np.log10(r_6200)
                        elif 'veil_r_6200' in x_names:
                            all_param[:, x_names.index('veil_r_6200')] = r_6200
                    if 'log_veil_r_9150' in x_names or 'veil_r_9150' in x_names:
                        r_9150 = self.exp.get_flux_at(wl, veiling, 9150.)/self.exp.get_flux_at(wl, all_obs[:,spec_locs], 9150.)
                        if 'log_veil_r_9150' in x_names:
                            all_param[:, x_names.index('log_veil_r_9150')] = np.log10(r_9150)
                        elif 'veil_r_9150' in x_names:
                            all_param[:, x_names.index('veil_r_9150')] = r_9150

                    if 'log_veil_r_5060' in x_names or 'veil_r_5060' in x_names:
                        r_5060 = self.exp.get_flux_at(wl, veiling, 5060., n_bins=15)/self.exp.get_flux_at(wl, all_obs[:,spec_locs], 5060., n_bins=15)
                        if 'log_veil_r_5060' in x_names:
                            all_param[:, x_names.index('log_veil_r_5060')] = np.log10(r_5060)
                        elif 'veil_r_5060' in x_names:
                            all_param[:, x_names.index('veil_r_5060')] = r_5060

                    if 'log_veil_r_6000' in x_names or 'veil_r_6000' in x_names:
                        r_6000 = self.exp.get_flux_at(wl, veiling, 6000., n_bins=15)/self.exp.get_flux_at(wl, all_obs[:,spec_locs], 6000., n_bins=15)
                        if 'log_veil_r_6000' in x_names:
                            all_param[:, x_names.index('log_veil_r_6000')] = np.log10(r_6000)
                        elif 'veil_r_6000' in x_names:
                            all_param[:, x_names.index('veil_r_6000')] = r_6000
                    
                    spec = new_
                    # all_obs[:, spec_locs] = self.exp.add_slab_veil(wl, all_obs[:,spec_locs], veil=veil_values, fslab_750=fslab_norm)
                else:
                    raise ValueError("Invalid configuration: use_Hslab_veiling=True & use_one_slab_grid!=True, but slab_grid is not specified.")
            else:
                spec = self.exp.add_veil(wl, spec, veil=veil_values)
                # all_obs[:, spec_locs] = self.exp.add_veil(wl, all_obs[:, spec_locs], veil=veil_values)
           
        # 2) Add extinction
        if extinct_flux:
            wl = self.exp.get_wl(self)[spec_indices]
            if "R_V" in x_names:
                Rv_array = all_param[:, x_names.index("R_V")]
            elif "R_V" in y_names:
                Rv_array = all_obs[:, y_names.index("R_V")]
            else:
                Rv_array = np.array(self.additional_kwarg["R_V"])
            spec = self.exp.extinct_spectrum(wl, spec, Av_array=all_param[:, x_names.index("A_V")], Rv_array = Rv_array )
            # all_obs[:, spec_locs] = self.exp.extinct_spectrum(wl, all_obs[:, spec_locs], Av_array=all_param[:, x_names.index("A_V")], Rv_array = Rv_array )
            
        # 여기서 플럭스 정리
        if use_spec:
            all_obs[:, spec_locs] = spec
        else:
            # when using flux values
            # now get median values for each flux values
            i_start = 0
            for i, loc in enumerate(spec_locs):
                # only using relevant spec_names
                i_end = i_start + len(spec_names_nested[i])
                all_obs[:, loc] = np.median( spec[:, i_start:i_end], axis=1 ) # use median flux 
                i_start = i_end

        
        # 3) Normalize flux: using total flux (small values), using mean flux (factor of len(y_names))
        if normalize_flux:
            if normalize_total_flux:
                all_obs[:, spec_locs] = all_obs[:, spec_locs]/np.sum(all_obs[:, spec_locs], axis=1).reshape(-1,1)
            elif normalize_mean_flux:
                all_obs[:, spec_locs] = all_obs[:, spec_locs]/np.mean(all_obs[:, spec_locs], axis=1).reshape(-1,1)
            elif normalize_f750:
                if use_spec==False:
                    if 'f7500' not in y_names:
                        raise ValueError("Cannot normalize to f7500: f7500 is not in y_names when using flux values.")
                    else:
                        f750_locs = y_names.index('f7500')
                        f750_array = all_obs[:, f750_locs]
                        all_obs[:, spec_locs] = all_obs[:, spec_locs] / f750_array.reshape(-1,1)
                else:
                    wl = self.exp.get_wl(self)[spec_indices]
                    f750_array = self.exp.get_dominika_f750_2d(wl, all_obs[:, spec_locs])
                    all_obs[:, spec_locs] = all_obs[:, spec_locs] / f750_array.reshape(-1,1)
                    
        # 2) Smoothing parameter 
        if smoothing:
            all_param = smooth_array(all_param, x_names, smoothing_sigma)
            
        # [TEST] Dummy slab: if True, change slab parameters to dummy values, for veil lower than veil_min
        # for additional informration. use "dummy_slab" dictionary in c.additional_kwarg
        # Do not use dummy when calculing rescale params!!!
        if dummy_slab:
            slab_params_all = ['Tslab','log_ne','log_tau0','log_Fslab']
            # slab params in x_names
            slab_params = list(set(x_names) & set(slab_params_all))
            if 'veil_r' in x_names:
                veil_values = all_param[:, x_names.index("veil_r")]
            elif 'log_veil_r' in x_names:
                veil_values =  10**all_param[:, x_names.index("log_veil_r")]
            else:
                raise ValueError("Cannot use dummy slab: neither veil_r nor log_veil_r is in x_names.")
            
            roi_dummy = veil_values < self.additional_kwarg["dummy_slab"]["veil_min"]
            dummy_smoothing = self.additional_kwarg["dummy_slab"].get("smoothing_factor", None)
            # smoothing factor: add noise to dummy value : dummy = mean + std*(outlier) + N(0, (smoothing_factor*std)^2)
            for param in slab_params:
                jj = x_names.index(param)
                dummy_val = self.mu_x[jj] + self.additional_kwarg["dummy_slab"]["outlier"]*(1/self.w_x[jj,jj]) # 10 sigma outlier
                if dummy_smoothing:
                    dummy_val = np.zeros(np.sum(roi_dummy))+dummy_val
                    dummy_val = smooth(dummy_val, (1/self.w_x[jj,jj])*dummy_smoothing )
                all_param[roi_dummy, jj] = dummy_val
            
        
        return all_param, all_obs
       
    # needed in __init__
    def update_rescale_params(self):
        
        # for randomizing parameters -> veiling / extinction
        veil_flux = False
        extinct_flux = False
        if self.random_parameters is not None:
            if "veil_r" in self.random_parameters.keys() or 'log_veil_r' in self.random_parameters.keys():
                veil_flux = True
            if "A_V" in self.random_parameters.keys():
                extinct_flux = True
        
        
        all_param, all_obs = self.load_data(smoothing = self.train_smoothing, smoothing_sigma = self.smoothing_sigma,
                                            obs_clipping = False,
                                            veil_flux = veil_flux, extinct_flux = extinct_flux,
                                            random_seed = 0,  
                                            normalize_flux=self.normalize_flux, 
                                            normalize_total_flux=self.normalize_total_flux, 
                                            normalize_mean_flux=self.normalize_mean_flux,
                                            normalize_f750=self.normalize_f750,
                                            )
        
        # update average, std
        self.mu_x = np.mean(all_param, 0)
        self.w_x = stddev_matrix(all_param)
        self.mu_y = np.mean(all_obs, 0)
        # self.w_y = whitening_matrix(all_obs ) 
        self.w_y = stddev_matrix(all_obs ) 
        
        if self.prenoise_training:
            if self.unc_corrl == 'Poisson': 
                all_lnsig = self.exp.calculate_poisson_ratio(all_obs, lum_min=1, log=True)
                #log(normalized sigma=sig/sigb)
                if self.unc_sampling == 'gaussian':  
                    self.mu_s = np.mean(all_lnsig, 0) + self.lsigb_mean
                    self.w_s = np.diag( 1/np.sqrt(np.std(all_lnsig, 0)**2. + self.lsigb_std**2.))
                elif self.unc_sampling == 'uniform':
                    self.mu_s = np.mean(all_lnsig, 0) + 0.5*(self.lsigb_min + self.lsigb_max)
                    self.w_s = np.diag( 1/np.sqrt(np.std(all_lnsig, 0)**2. + ((self.lsigb_max-self.lsigb_min)**2)/12  ))
            
            elif self.unc_corrl == 'Ind_Unif' or self.unc_corrl == 'Single':
                if self.unc_sampling == 'gaussian':
                    self.mu_s = np.zeros(len(self.y_names)) + self.lsig_mean
                    self.w_s = np.diag( 1/(np.zeros(len(self.y_names)) + self.lsig_std) )
                elif self.unc_sampling == 'uniform':
                    self.mu_s = np.zeros(len(self.y_names)) + 0.5*(self.lsig_min + self.lsig_max)
                    self.w_s = np.diag( 1/( np.zeros(len(self.y_names)) + np.abs(self.lsig_max-self.lsig_min)/np.sqrt(12) ) )
            elif self.unc_corrl == 'Ind_Man': # lsig_XX are read in list. 
                if self.unc_sampling == 'gaussian':
                    self.mu_s = np.zeros(len(self.y_names)) + np.array(self.lsig_mean)
                    self.w_s = np.diag( 1/(np.zeros(len(self.y_names)) + np.array(self.lsig_std)) )
                elif self.unc_sampling == 'uniform':
                    self.mu_s = np.zeros(len(self.y_names)) + 0.5*(np.array(self.lsig_min) + np.array(self.lsig_max))
                    self.w_s = np.diag( 1/( np.zeros(len(self.y_names)) + np.abs(np.array(self.lsig_max)-np.array(self.lsig_min))/np.sqrt(12) ) )
                    
            else:
                # just generate random lsig
                if self.unc_corrl=='Seg_Flux':
                    flux = all_obs[:, self.exp.get_spec_index(self.y_names, get_loc=True)]
                else: 
                    flux=None
                lsig_example = self.create_uncertainty(all_obs.shape, flux=flux)  
                self.mu_s = np.mean(lsig_example, 0)
                self.w_s = stddev_matrix(lsig_example)
                
            # elif self.unc_corrl == 'Seg_Flux':
            #     if self.unc_sampling == 'uniform':
            #         lsig_max = self.lsig_min + 3
            #         self.mu_s = np.zeros(len(self.y_names)) + 0.5*(self.lsig_min + lsig_max)
            #         self.w_s = np.diag( 1/( np.zeros(len(self.y_names)) + abs(lsig_max-self.lsig_min)/np.sqrt(12) ) )
           
            # elif self.unc_corrl == 'Seg_Unif': # all sig componets are sampled from the same probability=p(sigma) but wl is segmented
            #     if self.unc_sampling == 'uniform':
            #         self.mu_s = np.zeros(len(self.y_names)) + 0.5*(self.lsig_min + self.lsig_max)
            #         self.w_s = np.diag( 1/( np.zeros(len(self.y_names)) + abs(self.lsig_max-self.lsig_min)/np.sqrt(12) ) )
            #     elif self.unc_sampling == 'gaussian':  
            #         self.mu_s = np.zeros(len(self.y_names)) + self.lsig_mean
            #         self.w_s = np.diag( 1/(np.zeros(len(self.y_names)) + self.lsig_std) )
                    
        
        if self.use_flag:
            self.mu_f = np.zeros(len(self.flag_names)) + 0.5*(0 + 1)
            self.w_f = np.diag( 1/( np.zeros(len(self.flag_names)) + abs(1-0)/np.sqrt(12) ) )

        if self.wavelength_coupling:
            wl = self.exp.get_wl(self)[ self.exp.get_spec_index(self.y_names) ] # 아직 get_coupling_wavelength는 안써보기로
            data = np.random.choice(wl, len(wl)*1000)
            self.mu_wl = np.zeros(len(self.y_names)) + np.mean(data)
            self.w_wl =  np.diag( 1/( np.zeros(len(self.y_names)) + np.std(data) ) )
            
                    
    # differce to extract_ is this can preprocess , rescale the data if you want    
    def get_data(self, rawval=True, **kwarg):
        all_param, all_obs = self.load_data(**kwarg)
        if rawval==True:
            return all_param, all_obs
        if rawval==False:
            return self.params_to_x(all_param), self.obs_to_y(all_obs)
            
    
    def get_splitted_set(self, rawval=True, smoothing = False, smoothing_sigma = None, obs_clipping=False,
                         normalize_flux=False, normalize_total_flux=False, normalize_mean_flux=False, normalize_f750=False,
                         veil_flux = False, extinct_flux = False, random_seed=0, dummy_slab=False,
                         **kwargs):
        
        all_x, all_y = self.get_data(rawval=rawval, smoothing=smoothing,
                      smoothing_sigma=smoothing_sigma, obs_clipping=obs_clipping,
                      normalize_flux=normalize_flux, normalize_total_flux=normalize_total_flux, 
                      normalize_mean_flux=normalize_mean_flux, normalize_f750=normalize_f750,
                      veil_flux = veil_flux, extinct_flux = extinct_flux, random_seed = random_seed, dummy_slab=dummy_slab,
                      **kwargs )
        
        test_split = int(self.test_frac * len(all_x))
        np.random.seed(999)
        perm = np.random.permutation(len(all_x))#.astype(np.int32)
        all_x = all_x[perm]
        all_y = all_y[perm]
        
        test_set = [all_x[:test_split], all_y[:test_split], perm[:test_split]]
        train_set = [all_x[test_split:], all_y[test_split:], perm[test_split:]]
        
        return test_set, train_set
    
        
    # only follow the training setting. use only for train!
    def get_loaders(self, seed=0, param_seed = 0,):
              
        batch_size = self.batch_size
        rawval = False
        if self.prenoise_training==True:
            rawval = True
            
        # always follow config setting
        veil_flux = False
        extinct_flux = False
        if self.random_parameters is not None:
            if "veil_r" in self.random_parameters.keys() or 'log_veil_r' in self.random_parameters.keys():
                veil_flux = True
            if "A_V" in self.random_parameters.keys():
                extinct_flux = True
                
        dummy_slab = False
        if veil_flux==True and self.additional_kwarg is not None:
            if "dummy_slab" in self.additional_kwarg.keys():
                dummy_slab=True
           
                
        all_x, all_y = self.get_data(rawval=rawval, smoothing=self.train_smoothing,
                      smoothing_sigma=self.smoothing_sigma, obs_clipping=True,
                      normalize_flux=self.normalize_flux, normalize_total_flux=self.normalize_total_flux, 
                      normalize_mean_flux=self.normalize_mean_flux, normalize_f750=self.normalize_f750,
                      veil_flux = veil_flux, extinct_flux = extinct_flux, random_seed = param_seed, dummy_slab=dummy_slab,
                      )
 
        test_split = int(self.test_frac * len(all_x))
        
        np.random.seed(999)
        perm = np.random.permutation(len(all_x))#.astype(np.int32)   
        all_x = all_x[perm]
        all_y = all_y[perm]
        
        
        # Data augmentation for prenoise training and then rescale
        if self.prenoise_training==True:
            ymin = np.min(np.abs(all_y), axis=1) # minimum value for each 
            # unc_corrl, unc_sampling etc are considerd 
            if self.unc_corrl=='Seg_Flux':
                spec_loc = self.exp.get_spec_index(self.y_names, get_loc=True)
                if len(spec_loc)==0:
                    spec_loc = self.exp.get_flux_loc(self.y_names)
                flux = all_y[:, spec_loc]
            else: 
                flux=None
            lsig = self.create_uncertainty(all_y.shape, flux=flux)  # now expand, n_sig_MC, c.n_noise_MC are deprecated
            sig = 10**lsig
            # if c.n_sig_MC*c.n_noise_MC > 1:
            #     # N_mc expansion (n_noise_MC) + noise sampling (all line independent)
            #     x = torch.repeat_interleave(x, c.n_sig_MC*c.n_noise_MC, dim=0) 
            #     y = torch.repeat_interleave(y, c.n_sig_MC*c.n_noise_MC, dim=0)
            #     sig = torch.repeat_interleave(sig, c.n_noise_MC, dim=0)
            all_y = np.clip( all_y * (1+ sig*np.random.randn(*all_y.shape)), a_min=ymin.reshape(-1,1), a_max=None ) # all line independent
            
            # Transform to rescaled: np array again
            all_x = self.params_to_x(all_x)
            all_y = np.hstack( (self.obs_to_y(all_y), self.unc_to_sig(sig) ) )
        
        #-----------------------------------
        all_x = torch.Tensor(all_x)
        all_y = torch.Tensor(all_y)
        
        torch.manual_seed(seed)
        pin_memory = torch.cuda.is_available()
        
        
        if self.cond_net_code=="hybrid_cnn":
            # Need to divide spec_data (for cnn) and global_data (for global_net)
            roi_spec = self.exp.get_spec_index(self.y_names, get_loc=True, use_bool=True)
            if np.sum(roi_spec)==0:
                roi_spec = self.exp.get_flux_loc(self.y_names, use_bool=True)
            if self.prenoise_training==True:
                # divide y and sigma in axis=1
                all_y_3d = all_y.reshape(-1, 2, len(self.y_names)) 
                spec_data = all_y_3d[:, :, roi_spec] # (Models, channel, spectral points)
                global_data = all_y_3d[:, :, np.invert(roi_spec)].reshape(all_y.shape[0], -1) # (Models, global params, including their noises)
            else:
                spec_data = (all_y[:, roi_spec])[:,None,:]
                global_data = (all_y[:, np.invert(roi_spec)]).reshape(all_y.shape[0], -1)
                
            test_loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(all_x[:test_split], spec_data[:test_split], global_data[:test_split]),
                    batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=pin_memory, num_workers=get_num_workers())
            
            train_loader =  torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(all_x[test_split:], spec_data[test_split:], global_data[test_split:]),
                batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=pin_memory,  num_workers=get_num_workers())

        else: # fully connected network. basic cond_net
            test_loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(all_x[:test_split], all_y[:test_split]),
                    batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=pin_memory, num_workers=get_num_workers())
                 
            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(all_x[test_split:], all_y[test_split:]),
                batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=pin_memory,  num_workers=get_num_workers())
        
        
        return test_loader, train_loader
    
    
    def get_real_data(self, rawval=True,
                      normalize_flux=False, normalize_total_flux=False, normalize_mean_flux=False, normalize_f750 = False,):
        
        if self.prenoise_training:
            s_names = [k.replace('l','s') for k in self.y_names]
            rdata, sdata = self.exp.read_realdatabase(self.real_database, self.y_names, s_names=s_names) # read and only return usable y bins
        else:
            rdata = self.exp.read_realdatabase(self.real_database, self.y_names) # read and only return usable y bins
        
        # spec_indices = self.exp.get_spec_index(self.y_names)
        # Normalize flux: using total flux (small values), using mean flux (factor of len(y_names))
        if normalize_flux:
            if normalize_total_flux:
                rdata = rdata/np.sum(rdata, axis=1).reshape(-1,1)
            elif normalize_mean_flux:
                rdata = rdata/np.mean(rdata, axis=1).reshape(-1,1)
            elif normalize_f750:
                wl = self.exp.get_wl(self)[np.array([int(i[1:]) for i in self.y_names])]
                f750_array = self.exp.get_dominika_f750_2d(wl, rdata)
                rdata = rdata / f750_array.reshape(-1,1)
        
        if rawval == True:
            if self.prenoise_training:
                return (rdata, sdata)
            else:
                return rdata
        else:
            if self.prenoise_training:
                return (self.obs_to_y(rdata), self.unc_to_sig(sdata))
            else:
                return self.obs_to_y(rdata)
    
    def get_da_loaders(self):
        rawval = False
        if self.prenoise_training==True:
            rawval = True
            real_y, real_s = self.get_real_data(rawval=rawval,
                                    normalize_flux=self.normalize_flux, 
                                    normalize_total_flux=self.normalize_total_flux, 
                                    normalize_mean_flux=self.normalize_mean_flux, 
                                    normalize_f750 = self.normalize_f750,
                                    )
            
        else:
            real_y = self.get_real_data(rawval=rawval,
                                    normalize_flux=self.normalize_flux, 
                                    normalize_total_flux=self.normalize_total_flux, 
                                    normalize_mean_flux=self.normalize_mean_flux, 
                                    normalize_f750 = self.normalize_f750,
                                    )
        
        
        # reform the size: batch_size x real_frac
        n_rdb = len(real_y) # number in ddatabase
        n_real = int(self.batch_size * self.real_frac) # number requested
        
        if self.prenoise_training:
            if n_real == n_rdb:
                loader = ( torch.Tensor(real_y), torch.Tensor(real_s) )
            elif n_real > n_rdb: # make more
                n_need = n_real - n_rdb
                sel_ = np.random.choice(range(n_rdb), n_need)
                loader = ( torch.Tensor( np.vstack([real_y, real_y[sel_]]) ), torch.Tensor( np.vstack([real_s, real_s[sel_]]) ) )
            else: # chose
                sel_=np.random.choice(range(n_rdb), n_real)
                loader = ( torch.Tensor( real_y[sel_] ), torch.Tensor( real_s[sel_] ) )            
        else:
            if n_real == n_rdb:
                loader = torch.Tensor(real_y)
            elif n_real > n_rdb: # make more
                n_need = n_real - n_rdb
                loader = torch.Tensor( np.vstack([real_y, real_y[np.random.choice(range(n_rdb), n_need)]]) )
            else: # chose
                loader = torch.Tensor( real_y[np.random.choice(range(n_rdb), n_real)] )
            
        return loader
    

    """
    calculate_uncertainty is in expander
    
    size = tuple of (size[0], size[1], size[2],,,,)
    correlation (correlation between different sigmas in one obs): 
        - Ind_Man : no corr between sigmas, all different sampling range (but same sampling method)
                    HPs for sampling method are arrays
        - Ind_Unif : no corr between sigmas, the same sampling ftn for all sigmas
                    HPs for sampling method are constants
                    
    sampling method: gaussian, uniform ([min, max), but exchange=> (min, max] )
                                         
    def calculate_uncertainty(self, size, expand=1, correlation=None, sampling_method=None,
                            lsig_mean=None, lsig_std = None, lsig_min=None, lsig_max=None,):
    """    
  

    def create_uncertainty(self, size, expand=1, **kwarg):
        
        # kwarg = {}
        for att in ['lsig_min', 'lsig_max', 'lsig_mean', 'lsig_std']:
            kwarg[att] = getattr(self, att)
            
        if 'Seg' in self.unc_corrl:  # Seg_ option is only when using spectra. When using flux, use Ind_Unif
            # only for spectral y component
            # for all Seg cases: neccessary parameters are wl, y_names
            spec_indices = self.exp.get_spec_index(self.y_names)
            wl = self.exp.get_wl(self)[spec_indices]
            kw_seg = {'seg_size':getattr(self, 'wl_seg_size'),  'wl':wl, 
                      'y_names': getattr(self, 'y_names'),
                      'maxlgap':3 } # basic settup. you can update by sending kwarg
            kw_seg.update(kwarg)
            kwarg = kw_seg
            if self.unc_corrl=='Seg_Flux':
                if 'flux' not in kwarg.keys():
                    sys.exit("Flux is missing in random sigma sampling with Seg_Flux option")

            # flux 쓸때도 별도로 될 수 있도록 조치는 필요하긴 함.
            spec_locs = self.exp.get_spec_index(self.y_names, get_loc=True)
            if len(spec_locs)==0:
                spec_locs = self.exp.get_flux_loc(self.y_names, use_bool=False)
            if len(spec_locs) < len(self.y_names): 
                # non spectral/flux y components exist
                # check if special sampling is set for this value
                non_spec_locs = np.array([k for k in range(len(self.y_names)) if k not in spec_locs])
                noise_pdf = {}
                for param in [self.y_names[loc] for loc in non_spec_locs]:
                    if param+'_noise_pdf' in self.additional_kwarg.keys(): # 'R_V_noise_pdf
                        noise_pdf[param] = self.additional_kwarg[param+'_noise_pdf']
                        # this must contain 'sampling', and related 'lsig_min', 'lsig_max', etc
                    else:
                        # if not set, use the same pdf as spectral components
                        noise_pdf[param]={'sampling': self.unc_sampling}
                        for att in ['lsig_min', 'lsig_max', 'lsig_mean', 'lsig_std']:
                            noise_pdf[param][att] = kwarg[att]
                                
                    kwarg['noise_pdf']=noise_pdf
            
       
        return self.exp.calculate_random_uncertainty(size, expand=expand, 
                                                     correlation=self.unc_corrl, sampling_method=self.unc_sampling,
                                                     **kwarg)
         
    def create_random_flag(self, N_data):
        """
        Create random flag for training: flags = 0/1
        """
        size = (N_data, len(self.flag_names))
        
        return np.random.randint(low=0, high=2, size=size)

    def create_coupling_wavelength(self, N_data):
        """
        Create wavelength corresponding to the flux
        """
        wl = self.exp.get_wl(self)[self.exp.get_spec_index(self.y_names)]
        return np.repeat(wl.reshape(1,-1), N_data, axis=0)





