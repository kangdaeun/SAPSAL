#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 12:04:49 2026

@author: daeun

Fuctions for resimulation.

only for limited network versions

"""

import numpy as np
import sys
import pandas as pd
# When reading files in package (files are saved in data/)
from importlib import resources
import pickle
import gzip

import multiprocessing as mp
import astropy.units as units



from . import expander as exp
from . import HSlabModel
# from .expander import 

## Functions for file loading (files in data/)
def load_file_in_data(filename):
    """3.9+ 전용: sapsal/data/ 폴더 내 파일을 읽어 객체로 반환"""
    # 1. 리소스 위치 정의
    source = resources.files("sapsal.data").joinpath(filename)
    
    # 2. 안전한 경로 확보 및 처리
    with resources.as_file(source) as safe_path:
        suffix = safe_path.suffix.lower()
        
        # if safe_path.name.endswith('.gz'):
        #     with gzip.open(safe_path, 'rb') as f:
        #         return pickle.load(f)
        # if suffix == '.pkl':
        #     with open(safe_path, 'rb') as f:
        #         return pickle.load(f)

        # if suffix in ['.pkl', '.pkl.gz', '.pkl.xz']:
        #     return pd.read_pickle(safe_path)
        
        if suffix == '.npz': # These are FRAPPE grid
            # mmap_mode 등을 쓰지 않는 일반 로딩이라면 즉시 리턴 가능
            return np.load(safe_path)
        else:
            try: 
                return pd.read_pickle(safe_path)
            except:
                raise ValueError(f"Not supported extension: {suffix}")
                 
        
                
        



#----------------------------------------------------------------------------

def resampspec(wlsamp, wl, fl, err=None):
    """
    Function to resample an input spectrum on a new wavelength grid
    This is simple resampling using average of the bins. Not smoothing!
    Becareful not to change resolution much
    """
    bin_edges_midpoints = (wlsamp[:-1] + wlsamp[1:]) / 2.0
    
    if len(wlsamp) > 1:
        first_bin_start = wlsamp[0] - (wlsamp[1] - wlsamp[0]) / 2.0
    else: 
        first_bin_start = wlsamp[0] - 1.0 # 임의의 작은 값
    
    if len(wlsamp) > 1:
        last_bin_end = wlsamp[-1] + (wlsamp[-1] - wlsamp[-2]) / 2.0
    else:
        last_bin_end = wlsamp[0] + 1.0 # 임의의 작은 값

    bin_edges = np.concatenate(([first_bin_start], bin_edges_midpoints, [last_bin_end]))

    bin_indices = np.digitize(wl, bin_edges) - 1 # -1을 하여 0부터 시작하는 인덱스로 조정

    # 유효한 빈 인덱스 (0에서 len(wlsamp)-1 사이)만 필터링
    valid_indices = (bin_indices >= 0) & (bin_indices < len(wlsamp))
    
    # 3. 각 빈에 속하는 플럭스 값들의 합과 개수 계산
    # np.bincount는 정수 배열의 각 값의 출현 횟수를 세거나, 가중치를 적용하여 합을 계산합니다.
    
    # 플럭스 합계
    # minlength는 bincount가 계산할 최소 인덱스 범위를 설정하여, 빈 값이 0인 경우에도 해당 빈이 포함되도록 합니다.
    # len(wlsamp)가 최대 빈 인덱스가 되므로, len(wlsamp)를 지정합니다.
    flux_sum = np.bincount(bin_indices[valid_indices], weights=fl[valid_indices], minlength=len(wlsamp))
    
    # 각 빈에 속하는 데이터 포인트의 개수
    bin_counts = np.bincount(bin_indices[valid_indices], minlength=len(wlsamp))
    
    # 4. 평균 계산 (0으로 나누는 오류 방지)
    rfl = np.zeros(len(wlsamp))
    # bin_counts가 0보다 큰 빈에 대해서만 평균을 계산
    non_zero_bins = bin_counts > 0
    rfl[non_zero_bins] = flux_sum[non_zero_bins] / bin_counts[non_zero_bins]

    # 4. If error data is provided, calculate the resampled error (rerr)
    if err is not None:
        # Calculate the sum of squared errors for each bin
        err_sq_sum = np.bincount(bin_indices[valid_indices], weights=err[valid_indices]**2, minlength=len(wlsamp))
        
        # Initialize the resampled error array with zeros
        rerr = np.zeros(len(wlsamp))
        
        # Calculate the new error using the error propagation formula
        rerr[non_zero_bins] = np.sqrt(err_sq_sum[non_zero_bins]) / bin_counts[non_zero_bins]
        
        return rfl, rerr
    
    return rfl


# SpT-number conversions in FRAPPE 
def spt_coding(spt_in):
	# give a number corresponding to the input SpT
	# the scale is 0 at M0, -1 at K7, -8 at K0 (K8 is counted as M0),  -18 at G0
	if np.size(spt_in) == 1:
		if spt_in[0] == 'M':
			spt_num = float(spt_in[1:])
		elif spt_in[0] == 'K':
			spt_num = float(spt_in[1:])-8.
		elif spt_in[0] == 'G':
			spt_num = float(spt_in[1:])-18.
		elif spt_in[0] == 'F':
			spt_num = float(spt_in[1:])-28.
		elif spt_in[0] == 'A':
			spt_num = float(spt_in[1:])-38.
		elif spt_in[0] == 'B':
			spt_num = float(spt_in[1:])-48.
		elif spt_in[0] == 'O':
			spt_num = float(spt_in[1:])-58. # added by Da Eun
		elif spt_in[0] == 'L':
			spt_num = float(spt_in[1:])+10.
		elif spt_in[0] == '.':
			spt_num = -99.
		else:
			sys.exit('what?')
		return spt_num
	else:
		spt_num = np.empty(len(spt_in))
		for i,s in enumerate(spt_in):
			if s[0] == 'M':
				spt_num[i] = float(s[1:])
			elif s[0] == 'K':
				spt_num[i] = float(s[1:])-8.
			elif s[0] == 'G':
				spt_num[i] = float(s[1:])-18.
			elif s[0] == 'F':
				spt_num[i] = float(s[1:])-28.
			elif s[0] == 'A':
				spt_num[i] = float(s[1:])-38.
			elif s[0] == 'B':
				spt_num[i] = float(s[1:])-48.
			elif s[0] == 'O':
				spt_num[i] = float(s[1:])-58. # added by Da Eun
			elif s[0] == 'L':
				spt_num[i] = float(s[1:])+10.
			elif s[0] == '.':
				spt_num[i] = -99.
			else:
				sys.exit('what?')
		return spt_num
	

def convScodToSpTstring(scod):
	if np.size(scod) == 1:
		if scod<-18 or scod >10:
			print('out of bound')
			return None
		elif scod>=0:
			return 'M'+str(scod)
		elif scod<0 and scod>=-8:
			scodRet = 8+scod
			return 'K'+str(scodRet)
		elif scod<-8 and scod>-18:
			scodRed = 18+scod
			return 'G'+str(scodRed)
		return None
	else:
		spt_out = np.empty(len(scod),dtype = 'U64')
		for s,i in enumerate(scod):

			if i<-18 or i >10:
				print('out of bound')
				spt_out[s] = 'NaN'
				#return None
			elif i>=0:
				#return 'M'+str(i)
				spt_out[s] = 'M'+"%.1f" % (i)
			elif i<0 and i>=-8:
				iRet = 8+i
				spt_out[s] = 'K'+"%.1f" % (iRet)
			elif i<-8 and i>-18:
				iRed = 18+i
				spt_out[s] = 'G'+"%.1f" % (iRed)
			else: spt_out[s] ='NaN'
		return spt_out
	


############################################################################
# Resimulation related functions
############################################################################

PHOENIX_GRID_FILES = {
	'Settl': {
            'MUSE': "SynthSpecDF-BT-Settl_29072022.pkl.xz", # old Settl MUSE (K25): converted to xz
            'X-shooter': "XS_BT-Settl-CIFIST_Fb_SynthSpecDF_20260201.pkl.gz" # new Settl X-Shotter
           },
     
    'Dusty': {
		    'MUSE':"Restricted_mp_Dusty-last.SynthSpecDF_29092022.pkl.xz", # old Dusty MUSE (K25)
            'X-shooter':"XS_BT-Dusty_Fb_SynthSpecDF_20260201.pkl.gz" # new Dusty X-Shotter
            }, 

    'Settl_new': {
            'MUSE': "MUSE_BT-Settl-CIFIST_Fb_SynthSpecDF_20250904.pkl.gz", # new Settl MUSE 
            'X-shooter': "XS_BT-Settl-CIFIST_Fb_SynthSpecDF_20260201.pkl.gz" # new Settl X-Shotter
           },
     
    'Dusty_new': {
		    'MUSE':"MUSE_BT-Dusty_Fb_SynthSpecDF_20250904.pkl.gz", # new Dusty MUSE
            'X-shooter':"XS_BT-Dusty_Fb_SynthSpecDF_20260201.pkl.gz" # new Dusty X-Shotter
            },     
}

def prepare_phoenix_grid(grid_names = ['Settl', 'Dusty', 'Settl_new', 'Dusty_new'], wl_grid = 'X-shooter'):
    """
    Load appropirate Phoenix interpolation grid: XS/MUSE range, which library?
    grid_names: list of grid key names to load (related to PHOENIX library)
    wl_grid: str. 'MUSE' or 'X-shooter'

    return ASS_dic, which includes loaded interpolation grids, and db_range_dic


    현재 SpD, SpDx외의 경우 알아서 AAS_dic, db_range_dic을 제작하면 된다. 다만 문제는 다음 prepare param에 전달할 방법이 아직 없음.
    """
    # import pandas as pd
    # import gzip

    # db_pipe = main_dir + 'DB_pipeline/'
    
    ASS_dic = {}
    for name in grid_names:
        filename = PHOENIX_GRID_FILES[name][wl_grid]
        ASS_dic[name] = load_file_in_data(filename) # returned data is in pandas 
        if not isinstance(ASS_dic[name], pd.DataFrame):
            ASS_dic[name] = pd.DataFrame(ASS_dic[name])
			
        # if name=='Settl':
        #     if wl_grid=='MUSE':
        #         ASS_dic[name] = pd.read_pickle(db_pipe+"spectral_libraries/SynthSpecDF-BT-Settl_29072022.pkl") # old Settl MUSE
        #     elif wl_grid=='X-shooter':
        #         with gzip.open(db_pipe+"spectral_libraries/XS_BT-Settl-CIFIST_Fb_SynthSpecDF_20250828.pkl.gz", 'rb') as f:
        #             ASS_dic[name] = pd.read_pickle(f)
        # if name=='Dusty':
        #     if wl_grid=='MUSE':
        #         ASS_dic[name] = pd.read_pickle(db_pipe+"spectral_libraries/Restricted_mp_Dusty-last.SynthSpecDF_29092022.pkl") # old Dusty MUSE
        #     elif wl_grid=='X-shooter':
        #         with gzip.open(db_pipe+"spectral_libraries/XS_BT-Dusty_Fb_SynthSpecDF_20250828.pkl.gz", 'rb') as f:
        #             ASS_dic[name] = pd.read_pickle(f)
        # if name=='Settl_new':
        #     if wl_grid=='MUSE':
        #         with gzip.open(db_pipe+"spectral_libraries/MUSE_BT-Settl-CIFIST_Fb_SynthSpecDF_20250708.pkl.gz", 'rb') as f:
        #             ASS_dic[name] = pd.read_pickle(f)
        #     elif wl_grid=='X-shooter':
        #         with gzip.open(db_pipe+"spectral_libraries/XS_BT-Settl-CIFIST_Fb_SynthSpecDF_20250828.pkl.gz", 'rb') as f:
        #             ASS_dic[name] = pd.read_pickle(f)
        # if name=='Dusty_new':
        #     if wl_grid=='MUSE':
        #         with gzip.open(db_pipe+"spectral_libraries/MUSE_BT-Dusty_Fb_SynthSpecDF_20250708.pkl.gz", 'rb') as f:
        #             ASS_dic[name] = pd.read_pickle(f)
        #     elif wl_grid=='X-shooter':
        #         with gzip.open(db_pipe+"spectral_libraries/XS_BT-Dusty_Fb_SynthSpecDF_20250828.pkl.gz", 'rb') as f:
        #             ASS_dic[name] = pd.read_pickle(f)
    
    db_range_dic = {}
    for key, df in ASS_dic.items():
        db_range_dic[key]= {
                            'logG': (df['LogG'].min(), df['LogG'].max()),
                            'Teff': (df['Teff'].min(), df['Teff'].max()),
							'logTeff': (np.log10(df['Teff'].min()), np.log10(df['Teff'].max()))
                            }

    # db_range_dic['Settl']={
    #         'Teff':[2600, 7000],
    #         'logTeff':[np.log10(2600), np.log10(7000)],
    #         'logG': [2.5, 5],
    # }
    # db_range_dic['Dusty']={
    #     'Teff':[2600, 4000],
    #     'logTeff':[np.log10(2600), np.log10(4000)],
    #     'logG': [3.0, 5.5],
    # }
    # db_range_dic['Settl_new']={
    #     'Teff':[2600, 7000],
    #     'logTeff':[np.log10(2600), np.log10(7000)],
    #     'logG': [2.5, 5.5],
    # }
    # db_range_dic['Dusty_new']={
    #     'Teff':[2600, 4000],
    #     'logTeff':[np.log10(2600), np.log10(4000)],
    #     'logG': [3.0, 5.5],
    # }

    return ASS_dic, db_range_dic




def prepare_frappe_grid( wl_grid = 'X-shooter', only_range=False, verbose=False ):
    """
    Load appropirate FRAPPE grid: XS/MUSE range
    wl_grid: str. 'MUSE' or 'X-shooter'

    return cint (frappe class), spt_min (latest), spt_max (earliest) (spts are str)

    FRAPPE ver. 0.1. now in sapsal package.
    """
    # sys.path.append(frappe_path)
    # import PhotFeatures_Ray as pf
    
    from .FRAPPE import PhotFeatures_Ray as pf

    if wl_grid=='MUSE':
        # grid_path = main_dir + 'Models_Obs/FrappeDB/' + 'Grids_cINN/Tpl_G8_M9.5_MUSE_rad2.5_deg2_Nmc1000.npz' # MUSE 
        grid_name = 'Tpl_G8_M9.5_MUSE_rad2.5_deg2_Nmc1000.npz' # MUSE 
    elif wl_grid=='X-shooter':
        # grid_path = main_dir + 'Models_Obs/FrappeDB/' + 'Grids_cINN/Tpl_G8_M9.5_XSLowRes_rad2.5_deg2_Nmc1000.npz' # XS 
        grid_name = 'Tpl_G8_M9.5_XSLowRes_rad2.5_deg2_Nmc1000.npz' # XS 

    # cint = pf.classIII(grid_path)
    source = resources.files("sapsal.data").joinpath(grid_name)
    with resources.as_file(source) as safe_path:
        # pf.func11이 np.load 등을 실행하는 동안 safe_path(파일)가 유지됨
        cint = pf.classIII(safe_path)
        
    usedFeatures = cint.getUsedInterpFeat()
    wl = (usedFeatures[:,0]+usedFeatures[:,1])/2
    normWLandWidth = cint.getUsedNormWl() 
    if verbose:
        print("===== FRAPPE grid information =====")
        print("Used grid:", grid_name)
        print("Grid with %d spectral bins, wl: [%.3f, %.3f]nm with dwl=%g nm, normalized at %.f nm \n"%(len(usedFeatures), wl[0], wl[-1], np.unique(wl[1:]-wl[:-1])[0], normWLandWidth[0]))
    
    spt_code_in_grid= cint.sptCode
    spt_in_grid = np.array([pf.convScodToSpTstring(k) for k in spt_code_in_grid])
    spt_min = spt_in_grid[-1]; spt_max = spt_in_grid[0]
    if verbose:
        print('SpT in grid (%d):'%len(spt_in_grid), spt_in_grid)

    # if only spt range is needed
    if only_range:
        if verbose:
            print("===================================\n")
        return spt_min, spt_max
    
    # Modify NaN -> mean of +1, -1 points. boundary=neightbor value
    # Modify Neg (<0) -> mean of +1,-1 point if positive
    flux_matrix = cint.medInterp
    for i in range(flux_matrix.shape[1]):
        spec = flux_matrix[:,i]
        i_nan = np.where( np.isfinite(spec)==False )[0]
        if len(i_nan)>0:
            for ii in i_nan:
                if ii==0: new_val = spec[ii+1]
                elif ii+1==len(spec): new_val = spec[ii-1]
                else: new_val = np.nanmean(spec[ii-1:ii+1+1])
                flux_matrix[ii,i] = new_val
                # print(new_val, flux_matrix[ii-1,i], flux_matrix[ii+1, i])
    
        i_neg = np.where( spec < 0 )[0]
        if len(i_neg)>0:
            for ii in i_neg:
                if ii==0: new_val = spec[ii+1]
                elif ii+1==len(spec): new_val = spec[ii-1]
                else: new_val = np.nanmean([spec[ii-1], spec[ii+1]])
                flux_matrix[ii,i] = new_val
        if len(i_nan)+len(i_neg)>0:
            if verbose:
                print(f"{spt_in_grid[i]}: {len(i_nan)} NaN, {len(i_neg)} Neg cases modified")
    cint.medInterp = flux_matrix
    if verbose:
        print("===================================\n")

    return cint, spt_min, spt_max


def prepare_resim_params(param_table, wl_grid='X-shooter', grid_model='phoenix', phoenix_type='SpDx', 
                        #   use_FRAPPE=False, use_phoenix=True, use_SpDx=True, use_SpD=False, 
                          clip_logG = True, fixed_logg=4.0, lib_out_error=False,
                          verbose = False):
    """
    Get parameter table to run and process parameters runnable for resimulation. 
    param_table: parameters to resimulate. astropy table
    wl_grid: wl_grid option to load grids (X-shooter or MUSE)
    grid_model: 'phoenix' or 'frappe', 'phoenix_and_frappe'
        if frappe: param_table must include SpT (not SpTind)
        phoenix_and_frappe means combined set: library = -1 Frappe, else Phoenix, keep phoenix_type
    phoenix_type: 'SpD' or 'SpDx' (only when grid_model=='phoenix')
        param_table must include 'library'
    clip_logG: bool. if True, clip log g to be in the range of the model grid when log g is out of range. only for phoenix grid.
    fixed_logg: 4.0, This only applies for phoenix models, if log g is not in param_table 

    This does not actually use data in phoexn/frappe grids, only check the param ranges.
    
    added flags:
    flag_resim = 1 : can run resimulation / 0 : fail
    flag_settl = 1 : run based on Settl 
    flag_dusty = 1 : run based on Dusty
    flag_Tout = 1 : T or SpT is outof the range. -> this leads to flag_resim=0
    flag_Gout = 1 : log G is outof the range -> if clip_logG=True, than log g is clipped in resim_table. if not, flag_resim=0
    
    """

    resim_table = param_table.copy()

    if grid_model == 'phoenix_and_frappe':
        # Check libraray
        lib = np.around(resim_table['library'])
        roi_frappe = lib == -1
        roi_phoenix = np.logical_or(lib==0, lib==1)
        roi_out = np.invert(np.logical_or(roi_frappe, roi_phoenix))
        if np.sum(roi_frappe)+np.sum(roi_phoenix) != len(resim_table):
            if lib_out_error:
                raise ValueError("(grid_model=phoenix_and_frappe) Some libraries are neither Phoenix nor Frappe!")
            else:
                if verbose:
                    print("Warning: (grid_model=phoenix_and_frappe) Some libraries are neither Phoenix nor Frappe! They will be treated as Settl.")
                    resim_table['library'][roi_out] = 0 
        if np.sum(roi_frappe)>0: 
            # Prepare SpTind for frappe resimulation (SpT is string so..)
            resim_table['SpTind'] = np.zeros(len(resim_table))+np.nan
            resim_table['SpTind'][np.where(roi_frappe)[0]] = [exp.convert_temp_to_sptnum(10**lt, option='Tpl') for lt in resim_table['logTeff'][roi_frappe]]
    elif grid_model=='phoenix':
        roi_phoenix = np.ones(len(resim_table)).astype(bool)
        roi_frappe = np.invert(roi_phoenix)
    elif grid_model=='frappe':
        roi_frappe = np.ones(len(resim_table)).astype(bool)
        roi_phoenix = np.invert(roi_frappe)
    else:
        sys.exit("Currently only 'phoenix' and 'frappe' are supported for grid_model")
         
         
    # Set flag in advance
    resim_table['flag_Tout'] = np.zeros(len(resim_table)).astype(int) # 1 = T extrapolate
    resim_table['flag_resim'] = np.zeros(len(resim_table)).astype(int) # 1 = can run resimulation
    if grid_model == 'phoenix_and_frappe' or grid_model=='phoenix':
        resim_table['flag_Gout'] = np.zeros(len(resim_table)).astype(int) # 1 = log g extrapolate
        resim_table['flag_settl'] = np.zeros(len(resim_table)).astype(int) # 1: run based on Settl
        resim_table['flag_dusty'] = np.zeros(len(resim_table)).astype(int) # 1: run based on Dusty
    if grid_model == 'phoenix_and_frappe':
        resim_table['flag_frappe'] = roi_frappe.astype(int)
        resim_table['flag_phoenix'] = roi_phoenix.astype(int)

    # Set specific column needed for resimulation
    if np.sum(roi_frappe) > 0:
        resim_table['sptcode_list'] = np.zeros(len(resim_table)) + np.nan

    # Check log g available
    if np.sum(roi_phoenix) > 0 and 'logG' not in resim_table.colnames:
        resim_table['logG'] = np.zeros(len(resim_table)) + fixed_logg
        print(f"No log g data necessary for phoenix models requested -> Use fixed log g: {fixed_logg:g}")
    
    # if grid_model=='phoenix':
    if np.sum(roi_phoenix) > 0:
        final_idx = np.where(roi_phoenix)[0]
        resim_table_part = resim_table[final_idx]
        
        if verbose:
              print(f"Preparing resimulation parameters (PHOENIX, {wl_grid} wavelength grid)")
        if phoenix_type=='SpD':
            settl_name = 'Settl'
            dusty_name = 'Dusty'
            grid_names = [settl_name, dusty_name]
        elif phoenix_type=='SpDx':
            settl_name = 'Settl_new'
            dusty_name = 'Dusty_new'
            grid_names = [settl_name, dusty_name]
        else:
            sys.exit("Currently only SpD and SpDx types are supported for phoenix_type")

        
    
        ASS_dic, resim_db_range_dic = prepare_phoenix_grid(grid_names = grid_names, wl_grid = wl_grid)
        # only make one resim.
        # for dusty: if param not fit for dusty but fit for settl. then run settl (change roi_settl)  
        lib = resim_table_part['library']
        roi_settl = np.around(lib)==0; roi_dusty = np.around(lib)==1
                    
        settl_range = resim_db_range_dic[settl_name]
        dusty_range = resim_db_range_dic[dusty_name]
        
        # Check log g extrapolation
        roi_logg_out_settl = roi_settl * np.logical_or(resim_table_part['logG']>settl_range['logG'][1], resim_table_part['logG']<settl_range['logG'][0])
        roi_logg_out_dusty = roi_dusty * np.logical_or(resim_table_part['logG']>dusty_range['logG'][1], resim_table_part['logG']<dusty_range['logG'][0])
        roi_logg_out = np.logical_or(roi_logg_out_settl, roi_logg_out_dusty)
        # resim_table['flag_Gout'] = np.zeros(len(resim_table)).astype(int)
        resim_table['flag_Gout'][final_idx[roi_logg_out]] = 1
        
        if np.sum(roi_logg_out):
            if clip_logG:
                if verbose:
                    print("\tClip log g: %d models."%np.sum(roi_logg_out))
                resim_table['logG'][final_idx[roi_logg_out_settl]] = np.clip(resim_table_part['logG'][roi_logg_out_settl], a_max=settl_range['logG'][1], a_min=settl_range['logG'][0])
                resim_table['logG'][final_idx[roi_logg_out_dusty]] = np.clip(resim_table_part['logG'][roi_logg_out_dusty], a_max=dusty_range['logG'][1], a_min=dusty_range['logG'][0])
            else:
                # not clip -> cannot run resim
                roi_settl[roi_logg_out_settl] = False
                roi_dusty[roi_logg_out_dusty] = False

               
        # cannot make resim: T extrapolation
        roi_tout_settl = roi_settl*np.logical_or(resim_table_part['logTeff']>settl_range['logTeff'][1], resim_table_part['logTeff']<settl_range['logTeff'][0])
        roi_settl[roi_tout_settl] = False
        roi_tout_dusty = roi_dusty*np.logical_or(resim_table_part['logTeff']>dusty_range['logTeff'][1], resim_table_part['logTeff']<dusty_range['logTeff'][0])
        roi_dusty[roi_tout_dusty] = False
        # for dusty out case. if T in Settl. then change to settl (do not change library in resim_table)
        roi_tout_dusty_settl = roi_tout_dusty * np.logical_and(resim_table_part['logTeff']<=settl_range['logTeff'][1], resim_table_part['logTeff']>=settl_range['logTeff'][0])
        roi_settl[roi_tout_dusty_settl] = True
        # add flag
        # resim_table['flag_Tout'] = np.zeros(len(resim_table)).astype(int) # 1 means T out of range -
        resim_table['flag_Tout'][final_idx[roi_tout_dusty]] = 1 
        resim_table['flag_Tout'][final_idx[roi_tout_settl]] = 1
        resim_table['flag_Tout'][final_idx[roi_tout_dusty_settl]] = 0
        
        # (0.2, 0.8) as ambigous. & both param available
        # No more ambiguous
        
        roi_fail = np.invert(roi_settl)*np.invert(roi_dusty) # Neither Dusty nor Settl
        # check the number. each one is either Settl or Dusty
        if np.sum(np.invert(roi_fail)) != np.sum(roi_dusty)+np.sum(roi_settl):
            sys.exit("Settl, Dusty, and failure number mismatch.")
    
        resim_table['flag_resim'][final_idx] = np.invert(roi_fail).astype(int) # 1: can run resimulation / 0 : fail
        resim_table['flag_settl'][final_idx] = roi_settl.astype(int) # 1: run based on Settl
        resim_table['flag_dusty'][final_idx] = roi_dusty.astype(int) # 1: run based on Dusty
    
        if verbose:
            print(f"\tTotal {len(resim_table_part)} Phoenix models to run.")
            print(f"\t{np.sum(roi_fail)} models fail. Success: {np.sum(roi_settl)} Settl. {np.sum(roi_dusty)} Dusty.")

        # return resim_table

    # if grid_model=='frappe':
    if np.sum(roi_frappe) > 0:
        final_idx = np.where(roi_frappe)[0]
        resim_table_part = resim_table[final_idx]

        if verbose:
            print(f"Preparing resimulation parameters (FRAPPE, {wl_grid} wavelength grid)")
        # "spt_coding" function in FRAPPE code, now included in resim.py 
        spt_min, spt_max = prepare_frappe_grid(wl_grid=wl_grid, only_range=True, verbose=False) # read cint but no processing.
            
        # convert SpTind to SpT and Sptcode
        if 'SpT' not in resim_table_part.colnames:
            if 'SpTind' in resim_table_part.colnames:
                if verbose:
                    print("\tSpT not in param_table, converting SpTind to SpT")
                resim_table_part['SpT'] = [exp.convert_sptnum_to_spt(sn, out_nan=True, verbose=False) for sn in resim_table_part['SpTind'].data]
            else:
                sys.exit('Neither SpT nor SpTind in param_table')
            
        spt_list = resim_table_part['SpT'].data
        # Check NaN SpT: 'nan', '' or 'none' 'None' None -> weired SpTind values that cannot be converted to SpT 
        invalid_values = ['', np.nan, 'none', 'nan', 'None', None]
        roi_spt_invalid = np.isin(spt_list, invalid_values)
        # original FRAPPE spt_coding function accepts upto B-type. Now it can anyway convert O-type as well
        
        if verbose:
            print(f"\tInvalid SpT(SpTind) cases: {np.sum(roi_spt_invalid)} models")
        
        sptcode_list = np.array([spt_coding(spt) if not roi_spt_invalid[k] else np.nan for (k, spt) in enumerate(spt_list)]) # different to SpTind
        resim_table['sptcode_list'][final_idx] = sptcode_list
    
        # filter out extrapolated cases
        sptcode_min = spt_coding(spt_max); sptcode_max = spt_coding(spt_min)
    
        roi_spt_ext = np.logical_or(sptcode_list < sptcode_min, sptcode_list > sptcode_max) # this also filter out nan cases
        if verbose:
            print(f"\tSpT extrapolated cases: {np.sum(roi_spt_ext)} models")
        # resim_table['flag_Tout'] = np.zeros(len(resim_table)).astype(int)
        resim_table['flag_Tout'][final_idx[roi_spt_ext]] = 1
        resim_table['flag_Tout'][final_idx[roi_spt_invalid]] = 1
    
        roi_valid = np.invert(resim_table['flag_Tout'].data[final_idx].astype(bool))
        resim_table['flag_resim'][final_idx] = roi_valid.astype(int)
        if verbose:
            print(f"\tTotal {len(resim_table)} models to run.")
            print(f"\t{np.sum(np.invert(roi_valid))} models fail. Success: {np.sum(roi_valid)} models.")
        
        # return resim_table
    # else:
    #     sys.exit("Currently only 'phoenix' and 'frappe' are supported for grid_model")

        # If lib=-1, T>T_frappe (Tout) but T<7000
        if grid_model=='phoenix_and_frappe' and np.sum(roi_spt_ext) > 0:
            print('\t !!changed applied')
            roi_hot = (sptcode_list < sptcode_min) * np.invert(roi_spt_invalid) * (resim_table_part['logTeff'] <= 7000)
            if np.sum(roi_hot)>0:
                idx_hot = final_idx[roi_hot]
                resim_table['flag_resim'][idx_hot] = 1 
                resim_table['flag_settl'][idx_hot] = 1
                resim_table['flag_frappe'][idx_hot] = 0
                resim_table['flag_phoenix'][idx_hot] = 1
                if verbose:
                    print("\tChange FRAPPE-cases with T>G8 to Settl")

    
    return resim_table


def run_photosphere(resim_table,  wl_grid='X-shooter', grid_model='phoenix', phoenix_type='SpDx', 
                    chunk_size=5000, 
                    verbose = False, 
                  # use_multiprocessing = True, N_proc=-1, # multiprocessing only for phoenix
                  **kwarg):
    """ 
    Resimulate only photosphere spectra. For multiple models, sharing wavelength, grid, etc.
    Use parameters processed for resimulation by running prepare_resim_params function.

    wl_grid: wl_grid option to load grids (X-shooter or MUSE)
    grid_model: 'phoenix' or 'frappe', 'phoenix_and_frappe'
        phoenix_and_frappe means combined set: library = -1 Frappe, else Phoenix, keep phoenix_type
    phoenix_type: 'SpD' or 'SpDx' (only when grid_model=='phoenix')
        

    resampspec function is used to resample for phoenix cases.

    Return array of photosphere spectra. Shape: (N_models, N_wavelength)
    flux scale is different for each model. Do not compare different models before rescaling/normalization.
       
    """

    if grid_model == 'phoenix_and_frappe':
        if 'flag_frappe' not in resim_table.colnames or 'flag_phoenix' not in resim_table.colnames:
            # Check libraray
            lib = np.around(resim_table['library'])
            roi_frappe = lib == -1
            roi_phoenix = np.logical_or(lib==0, lib==1)
            if np.sum(roi_frappe)+np.sum(roi_phoenix) != len(resim_table):
                raise ValueError("(grid_model=phoenix_and_frappe) Some libraries are neither Phoenix nor Frappe!")
            
        roi_frappe = resim_table['flag_frappe'].data.copy().astype(bool)
        roi_phoenix = resim_table['flag_phoenix'].data.copy().astype(bool)
    elif grid_model=='phoenix':
        roi_phoenix = np.ones(len(resim_table)).astype(bool)
        roi_frappe = np.invert(roi_phoenix)
    elif grid_model=='frappe':
        roi_frappe = np.ones(len(resim_table)).astype(bool)
        roi_phoenix = np.invert(roi_frappe)
    else:
        sys.exit("Currently only 'phoenix' and 'frappe' are supported for grid_model")
         
   
    wl_muse = np.arange(4750.1572265625, 9351.4072265625, 1.25)
    wlmin = 3200; wlmax = 10000; dwl = 1
    wl_xshooter = np.concatenate( [np.arange(wlmin, np.floor(wl_muse[0]), dwl), wl_muse, np.arange(np.ceil(wl_muse[-1]), wlmax+dwl, dwl)] )

    if wl_grid=='MUSE': wl = wl_muse
    elif wl_grid=='X-shooter': wl = wl_xshooter

    n_obs = len(resim_table); n_lam = len(wl)
    raw_resim = np.zeros(shape=(n_obs, n_lam)) + np.nan

    if verbose:
        print("Resimulating photosphere...")

    # if grid_model=='phoenix':
    if np.sum(roi_phoenix) > 0:
        final_idx = np.where(roi_phoenix)[0]
        resim_table_part = resim_table[final_idx]

        from scipy.interpolate import RegularGridInterpolator

        if phoenix_type=='SpD':
            settl_name = 'Settl'
            dusty_name = 'Dusty'
            grid_names = [settl_name, dusty_name]
        elif phoenix_type=='SpDx':
            settl_name = 'Settl_new'
            dusty_name = 'Dusty_new'
            grid_names = [settl_name, dusty_name]
        else:
            sys.exit("Currently only SpD and SpDx types are supported for phoenix_type")
        
        ASS_dic, resim_db_range_dic = prepare_phoenix_grid(grid_names = grid_names, wl_grid = wl_grid)
       
        # resim_kwargs = {'normalization':None, 'logint_flux':False} # kwargs for SynthOptSpec (Not used anymore)
        roi_resim = resim_table_part['flag_resim']
        roi_settl = resim_table_part['flag_resim'] * roi_resim # Settl only runnable
        roi_dusty = resim_table_part['flag_dusty'] * roi_resim # Dusty only runnable
        if 'Teff' not in resim_table_part.colnames and 'logTeff' in resim_table_part.colnames:
            resim_table_part['Teff'] = 10**resim_table_part['logTeff']

        if verbose:
            print('\tRun Phoenix')

        # Assuming only settl and dusty for each
        for name, roi in zip([settl_name, dusty_name], [roi_settl.astype(bool), roi_dusty.astype(bool)]):
            if np.sum(roi)==0: # nothing to run
                continue
            idx_part = final_idx[roi]

            df = ASS_dic[name]
            flux = np.array([k for k in df['f']])
            coord_teff = np.unique(df['Teff'].values); coord_logg = np.unique(df['LogG'].values)
            params = (coord_logg, coord_teff)
            flux_grid = flux.reshape(len(coord_logg), len(coord_teff), flux.shape[-1])
            interpolator2d = RegularGridInterpolator(params, flux_grid, bounds_error=True)
            # Run and resample
            params_to_run = list(zip(resim_table_part['logG'].data[roi], resim_table_part['Teff'].data[roi]))
            _wl = df['wl'][0]
            if np.sum(roi) > chunk_size:
                 for i in range(0, len(params_to_run), chunk_size):
                    chunk_params = params_to_run[i : i + chunk_size]
                    chunk_indices = idx_part[i : i + chunk_size]
                    
                    chunk_ispec = interpolator2d(chunk_params)
                    for j, flux_val in enumerate(chunk_ispec):
                        raw_resim[chunk_indices[j]] = resampspec(wl, _wl, flux_val)
                    del chunk_ispec
            else:
                ispec = interpolator2d(params_to_run) # wl in grid is different to wl in use.
                for j, flux_val in enumerate(ispec):
                    raw_resim[idx_part[j]] = resampspec(wl, _wl, flux_val)
            # raw_resim[roi] = np.array([ resampspec(wl, _wl, flux) for flux in ispec] )

    # elif grid_model=='frappe':
    if np.sum(roi_frappe) > 0:
        final_idx = np.where(roi_frappe)[0]
        resim_table_part = resim_table[final_idx]

        if verbose:
            print('\tRun FRAPPE')
        cint, spt_min, spt_max = prepare_frappe_grid(wl_grid=wl_grid, verbose=verbose)
        usedFeatures = cint.getUsedInterpFeat()
        wl_frp = (usedFeatures[:,0]+usedFeatures[:,1])/2 # wl_grid is already applied when choosing frappe grid.
        if len(wl_frp) != len(wl):
             raise ValueError("FRAPPE wavelength does not match with wl set by wl_grid!!")

        # Run and make interpolated pure spectrum for MAP values
        sptcode_list = resim_table_part['sptcode_list'].data
        roi_resim = resim_table_part['flag_resim'].data
        N_samp = len(sptcode_list)
    
        # raw_resim = np.zeros(shape=(N_samp, len(wl)) ) + np.nan
        check_nan = np.zeros(N_samp)
        check_neg = np.zeros(N_samp)
        check_zer = np.zeros(N_samp)
        for i_model, sptcode in enumerate(sptcode_list):
            if roi_resim[i_model]:
                y, yerr = cint.getFeatsAtSpt_symetricErr(sptcode)
                raw_resim[final_idx[i_model]] = y
    
                # Before making DB, Check NaN, Neg, Zero cases
                roi_nan = np.isfinite(y)==False
                roi_neg = y < 0
                roi_zer = y == 0
                check_nan[i_model]= np.sum(roi_nan); check_neg[i_model] = np.sum(roi_neg); check_zer[i_model] = np.sum(roi_zer)
        if verbose:
            print(f"\tIn all resimulations, {np.sum(check_nan)} w/ NaN, {np.sum(check_neg)} w/ Neg, {np.sum(check_zer)} w/ Zero cases exist!!")

        # return raw_resim
    return raw_resim
    
    
    
### Veling and reddening

## For slab multiproessing
_shared_slab_kwargs = {}
# def _init_slab_worker(kwargs):
#     global _shared_slab_kwargs
#     _shared_slab_kwargs = kwargs

def _init_slab_worker(kwargs):
    global _shared_slab_kwargs
    # 만약 kwargs가 튜플이나 리스트에 담겨왔다면 첫 번째 요소를 꺼냄
    if isinstance(kwargs, (list, tuple)) and len(kwargs) > 0:
        _shared_slab_kwargs = dict(kwargs[0])
    else:
        _shared_slab_kwargs = dict(kwargs)

def _run_slab(task_row):
    """실제 계산만 수행하는 핵심 함수"""
    T, log_ne, log_tau0, wl = task_row
    # 전역에 저장된 설정값을 사용하여 통신 비용 최소화
    return HSlabModel.get_total_intensity(wl, T, 10**log_ne, 10**log_tau0, **_shared_slab_kwargs) 


def run_veiling_and_extinction(resim_table, phot_resim, wl_grid='X-shooter', 
                               run_veil=True, run_extinct=True, rv_values = None, return_veil=False,
                               use_Hslab_veiling=False, use_one_Hslab_model=False, 
                               slab_kwargs={"wl_sp": 7500, "Zi":1, "wl_sp_unit":units.AA, "include_Hn":True, "Int_lam":True, "lam_unit":units.AA},
                               use_multiprocessing=True, N_cpu=None, chunksize=None,
                               verbose=False):
    """
    Apply veiling and extinction to resimulated photosphere spectra.

    resim_table: astropy table including veiling and extinction parameters
    phot_resim: array of photosphere spectra. Shape: (N_models, N_wl)
    wl_grid: 'X-shooter' or 'MUSE'
    run_veil: bool. whether to apply veiling
    run_extinct: bool. whether to apply extinction
    rv_values: array or list of Rv values for extinction. If single value, apply to all models. If None, no extinction applied.
    return_veil: bool. whether to return veiling spectrum as well.
    use_Hslab_veiling: bool. whether to use Hslab model for veiling
    use_one_Hslab_model: bool. whether to use one Hslab model for all veiling values. Only when use_Hslab_veiling=True. (currently not used for the networks)
    slab_kwargs: dict. keyword arguments for HSlabModel.get_total_intensity function.
    use_multiprocessing: bool. whether to use multiprocessing for Hslab veiling calculation
    N_cpu: int or None. number of CPU to use for multiprocessing. If None, use all available CPUs.
    verbose: bool. whether to print progress.

    Output is dictionary always including spec_resim (the final)
    if veil applied, also includes veiled_resim
    if return_veil=True and veil applied, also includes veiling_resim
    if veiling_resim saved and extinction applied, also includes reddened_veiling_resim
    """
    outputs = {}

    wl_muse = np.arange(4750.1572265625, 9351.4072265625, 1.25)
    wlmin = 3200; wlmax = 10000; dwl = 1
    wl_xshooter = np.concatenate( [np.arange(wlmin, np.floor(wl_muse[0]), dwl), wl_muse, np.arange(np.ceil(wl_muse[-1]), wlmax+dwl, dwl)] )
    

    if wl_grid=='X-shooter':
        wl_resim = wl_xshooter
    elif wl_grid=='MUSE':
        wl_resim = wl_muse
    else:
         sys.exit("wl_grid should be either 'X-shooter' or 'MUSE'")
    # Check size of phot_resim
    if phot_resim.shape[1] != len(wl_resim):
        sys.exit("wl_grid and phot_resim wavelength size mismatch.")

    if run_veil:
        # check veiling values
        if 'veil_r' in resim_table.colnames:
            veil_values = resim_table['veil_r']
        elif 'log_veil_r' in resim_table.colnames:
            veil_values = 10**resim_table['log_veil_r'].data
        else:
            if verbose:
                print("Requested run_veil but no veil values in resim_table")
            run_veil = False 

    if verbose:
         print(f"Applying veiling and reddening for {len(resim_table)} models")

    spec_resim = phot_resim.copy()
    if run_veil:
        if use_Hslab_veiling==True:
            if use_one_Hslab_model==True: # This is currently set for only MUSE wavelength. No network trained on this setup.
                fslab_norm = exp.read_example_slab()
                spec_resim = exp.add_slab_veil(wl_muse, spec_resim, veil=veil_values, fslab_750=fslab_norm)
            else:
                if verbose:
                    print("\tRun slab veiling...") 
                # check all params are in resim_table
                # fslab_norm = np.zeros(shape=(len(resim_table), len(wl_resim)))
                # slab_param = ['Tslab','log_ne', 'log_tau0']

                t_arr = resim_table['Tslab'].data
                ne_arr = resim_table['log_ne'].data
                tau_arr = resim_table['log_tau0'].data

                if use_multiprocessing:
                    if N_cpu is None: N_cpu = mp.cpu_count()
                    else: N_cpu = np.min([N_cpu, mp.cpu_count()])

                    if chunksize is None:
                        # 수만 개일 때는 효율을 위해 100 이상, 작을 때는 1~20 사이로 자동 조절
                        chunksize = max(1, len(t_arr) // (N_cpu * 4))

                    # extract and prepare arguments for Hslab function
                    # tasks = [(t, ne, tau, wl_resim) for t, ne, tau in zip(t_arr, ne_arr, tau_arr)]
                    def task_generator():
                        for t, ne, tau in zip(t_arr, ne_arr, tau_arr):
                            yield (t, ne, tau, wl_resim)

                    with mp.Pool(processes=N_cpu, initializer=_init_slab_worker, initargs=(slab_kwargs,)) as pool:
                        # 수만 개 처리 시 효율을 위해 chunksize 적용
                        # chunk = max(1, len(tasks) // (N_cpu * 4))
                        # results = pool.map(_run_slab, tasks, chunksize=chunk)

                        results = pool.imap(_run_slab, task_generator(), chunksize=chunksize)
                        results = list(results) # needed to convert from iterator to list

                else:
                    if verbose:
                        print("\tRunning in single-processing mode...")
                    results = [HSlabModel.get_total_intensity(wl_resim, t, 10**ne, 10**tau, **slab_kwargs) for t, ne, tau in zip(t_arr, ne_arr, tau_arr)] 
                   
                results = np.array(results)
                # Normalization at 7500A
                f750_array = exp.get_f750(wl_resim, results)
                fslab_norm = results / f750_array.reshape(-1,1)
                
                # once fslab_norm is ready for all models
                veiled_resim, veiling_resim  = exp.add_slab_veil(wl_resim, spec_resim, veil=veil_values, fslab_750=fslab_norm, return_veil=True)
                  
        else: 
            print("\tApplying constant veiling...")
            # simple constant veiling
            veiled_resim = exp.add_veil(wl_resim, spec_resim, veil=veil_values )
            veiling_resim = veiled_resim - spec_resim

        spec_resim = veiled_resim
        outputs['veiled_resim'] = veiled_resim

        if return_veil:
            outputs['veiling_resim'] = veiling_resim    
    
    
    # Add reddening
    reddening = False
    if run_extinct:
        if rv_values is not None and 'A_V' in resim_table.colnames:
            rv_values = np.array(rv_values)
            av_values = resim_table['A_V'].data
            if rv_values.size == 1: # scalr or 1 value array
                rv_values = np.repeat(rv_values.item(), len(av_values))
            elif rv_values.size == len(av_values): # all different Rv 
                pass
            else:
                sys.exit("rv_values size mismatch with resim_table.")
            reddening = True   
        else:
            if verbose:
                print("\tRequested run_extinct but no extinction values in resim_table or no rv_values provided.") 
       
    if reddening:
        if verbose:
            print("\tApplying extinction...")
        spec_resim = exp.extinct_spectrum(wl_resim, spec_resim, Av_array=av_values, Rv_array = rv_values)
        outputs['spec_resim'] = spec_resim # final after extinction
        
        if 'veiling_resim' in outputs.keys():
            reddened_veiling_resim = exp.extinct_spectrum(wl_resim, outputs['veiling_resim'], Av_array=av_values, Rv_array = rv_values)
            outputs['reddened_veiling_resim'] = reddened_veiling_resim
    else:
        outputs['spec_resim'] = spec_resim # this will be the same as veiled_resim. if no veiling, same as phot_resim

    if verbose:
        print("Finished veiling and reddening")

    
    return outputs

def extract_resim_used(spec_resim, config):
    """
    Extract resimulated spectra only used in network. (with network foramt)
    spec_resim: array of resimulated spectra. shape (N_models, N_wl)
    config: network config object
    
    return: array of extracted spectra. shape (N_models, N_used)
    resim_used is the spectra still before normalization.
    """
    if spec_resim.ndim == 1: # single model
        spec_resim = spec_resim.reshape(1,-1)
    n_models = spec_resim.shape[0]

    # Extract data only used in network. (with network foramt)
    arg_ntf = exp.get_spec_index(config.y_names) # for spectrum based network
    if len(arg_ntf)==0:
        # spec_network = False
        # flux-based networks -> need to calculate median flux for normalization
        spec_names_nested = exp.get_spec_names_for_flux(config.y_names, wl=exp.get_wl(config), dwl=10) # bins to calculate median fluxes
        final_data = np.zeros(shape=(n_models,len(spec_names_nested)))+np.nan
        for i, names in enumerate(spec_names_nested):
            final_data[:, i] = np.median( spec_resim[:, [int(k[1:]) for k in names]], axis=1 )
        wl_data = np.array([float(config.y_names[k][1:]) for k in exp.get_flux_loc(config.y_names)])
    else:
        # spec_network = True
        final_data = spec_resim[:, arg_ntf] 
        wl = exp.get_wl(config) # for X-shooter, exp.get_wl(c) only gives blue+vis_muse. but resimulation includes red part. 
        # Here, we do not need red part, but just need wl that matches index 
        wl_data = wl[arg_ntf] 

    outputs = {}
    outputs['resim_used'] = final_data
    outputs['wl_used'] = wl_data

    return outputs

def get_normalize_factor(spec_resim, config, return_outputs=False, verbose=False):
    """
    Calculate normalization factor following network config.
    spec_resim: array of resimulated spectra. shape (N_models, N_wl)
    config: network config object
    wl_grid: 'MUSE' or 'X-shooter'
    
    return: array of normalization factors. shape (N_models,)

    if you just need normalization without saving normalzation factor, use exp.noramlize_flux function and extract_resim_used function together.
    """
    if config.normalize_flux != True:
        if verbose:
            print("The network did not use normalized flux.")
        return np.ones(spec_resim.shape[0])
    else:
        if verbose:
            print("Calculating normalization factors following network setup...")
        if spec_resim.ndim == 1: # single model
            spec_resim = spec_resim.reshape(1,-1)
        n_models = spec_resim.shape[0]

    arg_ntf = exp.get_spec_index(config.y_names) # for spectrum based network
    if len(arg_ntf)==0:
        spec_network = False
    else:
        spec_network = True

    outputs = extract_resim_used(spec_resim, config)
    final_data = outputs['resim_used']
    wl_data = outputs['wl_used']

    if config.normalize_total_flux:
        norm_factor = np.nansum(final_data, axis=1)
    elif config.normalize_f750:
        if spec_network:
            norm_factor = exp.get_f750(wl_data, final_data)
        else:
            sys.exit("Flux-based network does not use normalzation at 7500A. Something wrong.")
    elif config.normalize_mean_flux:
        norm_factor = np.nanmean(final_data, axis=1)
    else:
        norm_factor = np.ones(spec_resim.shape[0])

    # Not allow zero norm factor (to avoid error in division)
    norm_factor[norm_factor==0] = np.nan

    if n_models==1:
        norm_factor = norm_factor.flatten()

    if return_outputs:
        outputs['norm_factor'] = norm_factor
        return outputs
    else:
        return norm_factor


# Run all at once. Cannot change slab kwargs here. Simpliefied version. Not necessary.
def run_resimulation(param_table, wl_grid='MUSE', grid_model='phoenix', phoenix_type='SpDx', config=None,
                    return_all=False, return_veil=False, return_normalization=False, # 효율적인 셋업을 디폴트로
                    clip_logG=True, fixed_logg=4.0, # kwarg for prepare_resim_params
                    rv_values=None, use_multiprocessing=False, N_cpu=None,  # kwarg for run_veiling_and_extinction
                    use_Hslab_veiling=True, use_one_Hslab_model=False, run_veil=False, run_extinct=False, # if config is set, these will be controlled by config
                    verbose=False):
    """
    Run full resimulation: photosphere resimulation + veiling + extinction + normalization
    param_table: astropy table including all parameters needed for resimulation
    wl_grid: 'X-shooter' or 'MUSE'
    grid_model: 'phoenix' or 'frappe', 'phoenix_and_frappe'
    phoenix_type: 'SpD' or 'SpDx' (only when grid_model=='phoenix')
    config: network config object. if provided, run_veil, run_extinct, normalization (if return_normaliztion=True) will be controlled by config
            if return_normaliztion is True, config is necessary.

    return_all: bool. whether to return all intermediate outputs, turnign on return_veil, return_normaliztion.
                phot_resim, veiling_resim, veiled_resim, reddened_veiling_resim, spec_resim, resim_used, wl_used, norm_factor
    return_veil: bool. whether to return intermediate outpus from veiling: veiling_resim, veiled_resim, reddened_veiling_resim,
    return_normalization: bool. whether to return normalization factor and data used in network: resim_used, wl_used, norm_factor
        this function does not normalize the spectra. use norm_factor to normalize if needed.

    clip_logG: bool. whether to clip log g to be in the range of the model grid when log g is out of range. only for phoenix grid.
    rv_values: array or list of Rv values for extinction. If single value, apply to all models. 

    output is dictionary 
    default: resim_table, veiled_resim, spec_resim.

    
    """
    if return_all:
        return_veil = True
        return_normalization = True
       

    # Check running veiling and extinction
    if config is not None: 
        # use veiling and rv setup following network config
        if 'veil_r' in config.x_names or 'log_veil_r' in config.x_names:
            run_veil = True
            if config.use_Hslab_veiling: use_Hslab_veiling=True
            else: use_Hslab_veiling=False
            if config.use_one_Hslab_model: use_one_Hslab_model=True
            else: use_one_Hslab_model=False    
        else:
            run_veil = False
            
        if 'A_V' in config.x_names: run_extinct=True
        else: run_extinct=False

        if run_extinct:
            if 'R_V' in config.x_names: # network predicting Rv
                rv_values = param_table['R_V'].data
            elif 'R_V' in config.additional_kwarg.keys(): # network trained for fixed Rv
                rv_values = np.array([config.additional_kwarg['R_V']])
    
    # check basics
    if run_veil:
        if 'veil_r' not in param_table.colnames and 'log_veil_r' not in param_table.colnames:
            sys.exit("Veiling needed but no veil values in param_table")
    if run_extinct:
        if 'A_V' not in param_table.colnames or rv_values is None:
            sys.exit("Reddening needed but eith Av or Rv values missing")
    if return_normalization:
        if config is None:
            sys.exit("Normalization requested but no config provided.")


    # start with preparing resim_table
    resim_table = prepare_resim_params(param_table, wl_grid=wl_grid, grid_model=grid_model, phoenix_type=phoenix_type, 
                                       clip_logG=clip_logG, fixed_logg=fixed_logg, verbose=verbose) 
    # Run photosphere
    phot_resim = run_photosphere(resim_table, wl_grid=wl_grid, grid_model=grid_model, verbose=verbose)
    
   # veiling and reddening
    output = run_veiling_and_extinction(resim_table, phot_resim, wl_grid=wl_grid,
                                run_veil=run_veil, run_extinct=run_extinct, rv_values = rv_values, return_veil=return_veil,
                                use_Hslab_veiling=use_Hslab_veiling, use_one_Hslab_model=use_one_Hslab_model, 
                                use_multiprocessing=use_multiprocessing, N_cpu=N_cpu,
                                verbose=verbose)
    
    output['resim_table'] = resim_table    
    if return_all:  # add outputs for all processes
        output['phot_resim'] = phot_resim 
        # veiling is already added by return_veil=True
    
    if return_normalization:
        if config.normalize_flux: # only when the network is trained on normalized flux
            add_output = get_normalize_factor(output['spec_resim'], config, return_outputs=True, verbose=verbose)
            # get data used in network as well by return_outputs=True
            output.update(add_output) # add resim_used, wl_used, norm_factor

    print("Finished resimulation.\n")

    return output