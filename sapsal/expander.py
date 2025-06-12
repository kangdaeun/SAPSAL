import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
import os
from astropy.io import ascii
if sys.version_info >= (3, 9):
    import importlib.resources as resources
else:
    import pkgutil
# float_dtype = np.float64
# int_dtype = np.int64
# from astopy.table import Table

def read_database(tablename):
    
    whole_table = pd.read_csv(tablename)
    # whole_table = pd.read_csv('ecogal_spectra_training_data.csv')
    # whole_table = ascii.read( tablename, delimiter="\t", format="commented_header")    
    return whole_table


def divide_xy(table, x_names, y_names, random_parameters=None, random_seed=0, f_min_dic={}, f_max_dic={}):
    
    np.random.seed(int(random_seed))
    N = len(table)
    table_cols = set(table.columns)

    rparams = set(random_parameters.keys()) if random_parameters else set()

    # 전체 param/spec을 NaN으로 초기화
    params = np.full((N, len(x_names)), np.nan)#, dtype=np.float32)
    spec = np.full((N, len(y_names)), np.nan)#, dtype=np.float32)

    # === X 처리 ===
    for i, param in enumerate(x_names):
        if param in rparams:  # 1순위: 랜덤 파라미터
            mini, maxi = random_parameters[param]
            f_min = f_min_dic.get(param, 0.) # 0 if f_min is not specified for this parameter
            f_max = f_max_dic.get(param, 0.)
            params[:, i] = generate_random_parameter(N, min_value=mini, max_value=maxi, f_min=f_min, f_max=f_max)
        elif param in table_cols:  # 2순위: 테이블에서 가져오기
            params[:, i] = table[param].values
        # else: 그대로 nan

    # === Y 처리 ===
    for i, param in enumerate(y_names):
        if param in rparams:
            mini, maxi = random_parameters[param]
            f_min = f_min_dic.get(param, 0.)
            f_max = f_max_dic.get(param, 0.)
            spec[:, i] = generate_random_parameter(N, min_value=mini, max_value=maxi, f_min=f_min, f_max=f_max)
        elif param in table_cols:
            spec[:, i] = table[param].values
        # else: 그대로 nan

    return (params, spec)


# def divide_xy(table, x_names, y_names, random_parameters=None, random_seed=0, f_min_dic=None, f_max_dic=None,
#               exclude = []):
    
#     table_cols = table.columns.values.tolist()
#     # N_data = len(table)
#     # params = np.zeros( (N_data, len(x_names)) ) + np.nan
#     # spec = np.zeros( (N_data, len(y_names))) + np.nan
    
#     # all_in_table = set(x_names + y_names).issubset(table_cols) # all x_names and y_naems are in table

#     if random_parameters is None:
#         params = table.loc[:, x_names].values # if all x_names in table already, just read
#         spec = table.loc[:, y_names].values
        
#     else:
#         rparams = list(random_parameters.keys())
#         xn_in_rp = [k for k in x_names if k in rparams ]
#         yn_in_rp = [k for k in y_names if k in rparams ]
        
#         xn_nin_rp = [k for k in x_names if k not in rparams ]
#         yn_nin_rp = [k for k in y_names if k not in rparams ]
        
#         params = table.loc[:, xn_nin_rp]
#         spec = table.loc[:, yn_nin_rp]
        
#         np.random.seed(int(random_seed))
        
#         if len(xn_in_rp)>0:
#             for param in xn_in_rp:
#                 mini, maxi = random_parameters[param]
#                 f_min, f_max = 0., 0.
#                 try: 
#                     f_min = f_min_dic[param]
#                 except: 
#                     pass
#                 try:
#                     f_max = f_max_dic[param]
#                 except:
#                     pass
                
#                 ii = x_names.index(param)
#                 rval =  generate_random_parameter(len(table), min_value=mini, max_value=maxi,
#                                                                f_min=f_min, f_max=f_max)
#                 params.insert(ii, param, rval)
                
#         if len(yn_in_rp)>0:
#             for param in yn_in_rp:
#                 mini, maxi = random_parameters[param]
#                 f_min, f_max = 0., 0.
#                 try:
#                     f_min = f_min_dic[param]
#                 except:
#                     pass
#                 try:
#                     f_max = f_max_dic[param]
#                 except:
#                     pass
                
#                 ii = y_names.index(param)
#                 rval =  generate_random_parameter(len(table), min_value=mini, max_value=maxi,
#                                                                f_min=f_min, f_max=f_max)
#                 spec.insert(ii, param, rval)
                
#         params = params.values
#         spec = spec.values

#     return (params, spec)


def normalize_flux(obs_data, normalize_total_flux=None, normalize_mean_flux=None, normalize_f750=None, wl=None,):
    if normalize_total_flux:
        return obs_data/np.sum(obs_data, axis=1).reshape(-1,1)
    elif normalize_mean_flux:
        return obs_data/np.mean(obs_data, axis=1).reshape(-1,1)
    elif normalize_f750:
        f750_array = get_f750(wl, obs_data)
        return obs_data / f750_array.reshape(-1,1)
    else:
        return obs_data
    
def generate_random_parameter(N_data, min_value = None, max_value = None, 
                              f_min = 0., f_max = 0.
                              ):
    """
    generate random values from a uniform distribution using given min_vaule and max_value U(min_value, max_value)

    Parameters
    ----------
    N_data : int
        number of data.
    min_value : int, necessary
    max_value : int, necessary
    f_min : float, optional, default=0
        fraction of data to change random values to min_value
    f_max : float, optional, default=0.
        fraction of data to caange random values to max_value
        f_min + f_max < 1
    Returns
    -------
    np array randomly sampled from U(min_value, max_value)

    """
    values = min_value + np.random.rand(N_data) * (max_value - min_value)
    
    if f_min > 0 or f_max > 0:
        min_split = 0; max_split = 0
        if f_min > 0: min_split = int(N_data * f_min)
        if f_max > 0: max_split = int(N_data * f_max)
        
        # min from start. max from end
        values[:min_split] = min_value
        values[N_data-max_split:] = max_value

    return values

def get_spec_index(y_names, get_loc=False):
    # filter out indices for spectral bins : l2, l231, from y_names:
    # not the index in y_names, it changes 'l0' -> 0
        
    # [s for s in c.y_names if s.startswith('l') and s[1:].isdigit()][-1]
    
    if get_loc:
        return np.array([i for i, s in enumerate(y_names) if s.startswith('l') and s[1:].isdigit()])
    else:
        return np.array([int(s[1:]) for i, s in enumerate(y_names) if s.startswith('l') and s[1:].isdigit()]) 

def cardelli_extinction(wave, Av, Rv):
    # If you use it to apply a reddening to a spectrum, multiply it for the result of
    # this function, while you should divide by it in the case you want to deredden it.
    
    # here Av and Rv should be scalr values

    #ebv = Av/Rv

    x = 10000./ wave # Convert to inverse microns (wave is in Anstrom)
    npts = len(x)
    a = np.zeros(npts)
    b = np.zeros(npts)
    # extinction valid range: upto 3.3333 um, down to 3030A
    #******************************

    good = (x > 0.3) & (x < 1.1) #Infrared: 9090.90 A ~ 3.3333um
    Ngood = np.count_nonzero(good == True)
    if Ngood > 0:
        a[good] = 0.574 * x[good]**(1.61) # from Cardelli+1989
        b[good] = -0.527 * x[good]**(1.61)

    #****************************** 
    good = (x >= 1.1) & (x < 3.3) #Optical/NIR: 3030.30A ~ 9090.90A
    Ngood = np.count_nonzero(good == True)
    if Ngood > 0: #Use new constants from O'Donnell (1994)
        y = x[good] - 1.82
        c1 = [-0.505, 1.647, -0.827, -1.718, 1.137, 0.701, -0.609, 0.104, 1.0] #New coefficients
        c2 = [3.347, -10.805, 5.491, 11.102, -7.985, -3.989, 2.908, 1.952, 0.0] #from O'Donnell (1994)

        a[good] = np.polyval(c1,y)
        b[good] = np.polyval(c2,y)

    A_lambda = Av * (a + b/Rv) # dim(a) = dim(wave), so in this case, 

    ratio = 10.**(-0.4*A_lambda)

    return ratio # dim(ratio) = dim(wave)

    



def extinct_spectrum(wl, flux, Av_array, Rv_array):
    """
    wl must be 1D array
    Av_array and Rv_array can be one value or 1D array, in that case, flux should 1D, 2D respectively
    
    """
    
    single_data = True
    multi_Rv = False; multi_Av=False
    # check dimension of given Av and Rv
    
    if len(np.array(Rv_array).shape)==1 and np.array(Rv_array).shape[0] == np.array(flux).shape[0]:
        multi_Rv = True
        single_data = False
   
    if len(np.array(Av_array).shape)==1 and np.array(Av_array).shape[0] == np.array(flux).shape[0]:
        multi_Av = True
        single_data = False
       
        
    if single_data:
        return flux * cardelli_extinction(wl, Av_array, Rv_array)
    
    else:
        if not multi_Av:
            Av_array = np.repeat(Av_array, flux.shape[0])
        if not multi_Rv:
            Rv_array = np.repeat(Rv_array, flux.shape[0])
        
        for i in range(flux.shape[0]):
            flux[i, :] = flux[i, :] * cardelli_extinction(wl, Av_array[i], Rv_array[i])
        
        return flux



def get_f750(wl, fl):
    """
    wl: 1D array. apply to all obs
    fl: 1D array (one spec) or  2D array (N_obs)

    """
    id750 = np.abs(wl - 7500.).argmin()

    if fl.ndim == 1:
        return np.nanmedian(fl[id750 - 3:id750 + 3]) 

    elif fl.ndim == 2:
        sub_fl = fl[:, id750 - 3:id750 + 3]  # shape (N, 6)
        sorted_sub = np.sort(sub_fl, axis=1)
        median_vals = 0.5 * (sorted_sub[:, 2] + sorted_sub[:, 3])
        return median_vals

    else:
        raise ValueError("fl must be 1D or 2D numpy array")



# def get_dominika_f750(wl, fl):
#     id750 = np.abs(wl - 7500.).argmin()
#     f750 = np.nanmedian(fl[id750 - 3:id750 + 3]) # this is value
#     return f750

# def get_dominika_f750_2d(wl, fl): # 1D wavelength value, and 2D flux valu
#     """
#     wl: 1D wavelengh, applyt to all 
#     fl: 2D flux value: N_obs x N_spectral bins
#     """
#     # assume that flux is in 2D array (N_data, spec) and all have the same spectral bins
#     id750 = np.abs(wl - 7500.).argmin()
#     f750_2d = np.nanmedian(fl[:, id750 - 3:id750 + 3], axis=1) # this is 1D array
#     return f750_2d

# def get_dominika_f750_2d(wl, fl):
#     id750 = np.abs(wl - 7500.).argmin()
#     sub_fl = fl[:, id750 - 3:id750 + 3]  # shape: (N, 6)

#     # sort along last axis (axis=1)
#     sorted_sub = np.sort(sub_fl, axis=1)

#     # 직접 nan 처리 없이 중간값 추출: 중앙 두 값의 평균
#     # (6개 값 중 중간 두 개가 2번째와 3번째 값이므로)
#     median_2d = 0.5 * (sorted_sub[:, 2] + sorted_sub[:, 3])
#     return median_2d
 

# Veiling   
def add_veil(wl, fl, veil):
   
    f750 = get_f750(wl, fl)
    # for only one spectrum
    if fl.ndim == 1:
        fl = fl + veil*f750
    elif fl.ndim == 2: # for multiple spectrum that shares  wavelength and spectral bin but different veiling value
        if len(veil)==fl.shape[0]:
            fl = fl + (veil * f750).reshape(-1,1)
        else:
            ValueError("len(veil)!=N_obs")
    else:
        raise ValueError("fl must be 1D or 2D numpy array")
        
    return fl

# HSlab veiling 
def add_slab_veil(wl, fl, veil, fslab_750):
    f750 = get_f750(wl, fl)
    # for only one spectrum
    if fl.ndim == 1:
        fl = fl + veil * fslab_750 * f750
    
    elif fl.ndim == 2: # for multiple spectrum that shares  wavelength and spectral bin but different veiling value
        # one slab or different slabs
        # if fslab_750.ndim==1: # one slab for all spec -> make 2D array
        #     fslab_750 = np.repeat(fslab_750.reshape(1,-1), fl.shape[0], axis=0)
        if len(veil)==fl.shape[0]:
            # fl = fl + ((veil * f750).reshape(-1,1) )*fslab_750
            fl = fl + (veil * f750)[:, None] * fslab_750
        else:
            ValueError("len(veil)!=N_obs")
    else:
        raise ValueError("fl must be 1D or 2D numpy array")
    return fl

# read already saved one slab
def read_example_slab():
    package = "sapsal"
    filename = "Slab_T7_N13_t1_n750.txt"
    if sys.version_info >= (3, 9):  # Python 3.9 
        with resources.files(package).joinpath(filename).open("r", encoding="utf-8") as f:
            float_array = np.array(f.read().split(), dtype=float)
    else: 
        data = pkgutil.get_data(package, filename)
        float_array = np.array(data.decode("utf-8").split(), dtype=float)
    # filepath = "./Slab_T7_N13_t1_n750.txt" # normalized at 7500A
    # np.loadtxt(filepath)
    return float_array


def assign_slab_grid(slab_grid_file, params, x_names, y_names, random_seed=0):
    
    np.random.seed(int(random_seed))
    
    N = len(params)
    
    # 1. slab_grid 불러오기
    slab_grid = pd.read_csv(slab_grid_file)
    
    # 2. slab_grid에서 각 행마다 무작위 샘플 하나씩 선택 (슬라이스로)
    slab_idxs = np.random.randint(0, len(slab_grid), size=N)
    matched_slab = slab_grid.iloc[slab_idxs].reset_index(drop=True)
    
    # 3. slab_norm 생성
    spec_y_names = list(np.array(y_names)[get_spec_index(y_names, get_loc=True)])
    slab_norm = matched_slab[spec_y_names].values
    
    # 4. np.nan 위치 중 slab_grid에 있는 값으로 채우기
    slab_cols = set(slab_grid.columns)
    x_fill_names = list(set(x_names) & set(slab_cols))
    
    # X 파라미터 채우기
    for name in x_fill_names:
        i = x_names.index(name)
        col_data = params[:, i]
        if np.all(np.isnan(col_data)):
            params[:, i] = matched_slab[name].values
        elif np.any(np.isnan(col_data)):
            raise ValueError(f"Column '{name}' in params has partial NaNs, which is unexpected.")
    
    return params, slab_norm


    
    


def get_muse_wl():
    return np.arange(4750.1572265625, 9351.4072265625, 1.25) #from f20 (3681)
    

def get_coupling_wavelength(y_names):
    return get_muse_wl()[np.array([int(k[1:]) for k in y_names])]
    # wl = get_muse_wl()[np.array([int(i[1:]) for i in y_names])]
    # return np.repeat(wl.reshape(1,-1), N_data, axis=0)
    
# for domain adaptation 
def read_realdatabase(tablename, y_names, s_names=None): # read and only return usable y bins
    real_table = pd.read_csv(tablename)
  
    if len(y_names) < 3681:
        spec = real_table.loc[:, y_names].values
    else:
        spec = real_table.loc[:, ['l{:d}'.format(i) for i in range(3681)]].values
    
    if s_names is not None:
        sigma = real_table.loc[:, s_names].values
        return (spec, sigma)
    else:
        return spec


"""
 SpT related functions
"""
# SpT - SpT index 
def convert_spt_to_num(spt):
    if 'M' in spt:
        return float(spt[1:])+8 # K type: K0 ~ K7
    elif 'K' in spt:
        return float(spt[1:])
    elif 'G' in spt:
        return float(spt[1:])-10
    elif 'F' in spt:
        return float(spt[1:])-20
    elif 'A' in spt:
        return float(spt[1:])-30
    elif 'B' in spt:
        return float(spt[1:])-40
    elif 'O' in spt:
        return float(spt[1:])-50
    else:
        print("No matching")
        return None
                    
def convert_sptnum_to_spt(spt):
    if spt >=8:
        return 'M'+str(spt-8)
    elif spt <8 and spt >=0:
        return 'K'+str(spt)
    elif spt < 0 and spt >= -10:
        return 'G'+str(spt+10)
    elif spt < -10 and spt >= -20:
        return 'F'+str(spt+20)
    elif spt < -20 and spt >= -30:
        return 'A'+str(spt+30)
    elif spt < -30 and spt >= -40:
        return 'B'+str(spt+40)
    elif spt < -40 and spt >= -50:
        return 'O'+str(spt+50)
    else:
        print("No matching")
        return None
        

module_path = os.path.dirname(os.path.abspath(__file__))
cvt_file = module_path +'/' + 'SpT-Teff_combine.txt'
spt_convert_table = ascii.read(cvt_file , format='commented_header', delimiter='\t')
spt_option_dic = {'KH95': 'Teff_KH95', 'L03': 'Teff_L03', 'HH14': 'Teff_HH14', 
              'Tr14_ori': 'Teff_Tr14_It24', 'Tpl_ori': 'Teff_Templates', 
              'Tr14': 'Tr14_ext', 'Tpl': 'Tpl_ext'}
spt_desc_dic = {}
for key, name in spt_option_dic.items():
    teff = spt_convert_table[name]
    tab = spt_convert_table[np.isfinite(teff)]
    late = tab['SpT'][np.argmin(tab[name])]
    early = tab['SpT'][np.argmax(tab[name])]
    spt_desc_dic[key] = "%s (%s - %s)"%(key, early, late)

def convert_temp_to_sptnum(teff, option='Tpl', out_nan=False):

    key = spt_option_dic[option]
      
    ref_temp = spt_convert_table[key]
    ref_sptind = spt_convert_table['SpTind']
    roi_valid = np.isfinite(ref_temp)
    ref_temp = ref_temp[roi_valid]
    ref_sptind = ref_sptind[roi_valid]
    
    i_tmax = np.argmax(ref_temp)
    i_tmin = np.argmin(ref_temp)
    
    if teff < np.min(ref_temp):
        if out_nan: return np.nan
        else: sys.exit("Out of the range: %s"%spt_desc_dic[option])
        # return ref_sptind[i_tmin]
    if teff > np.max(ref_temp):
        if out_nan: return np.nan
        else: sys.exit("Out of the range: %s"%spt_desc_dic[option])
        # return ref_sptind[i_tmax]
    
    arg_min = np.argmin(abs(ref_temp-teff))
    if ref_temp[arg_min] >= teff:
        ind_right = arg_min # higher temp
        ind_left = arg_min +1 # sort from High to Low Teff
    else:
        ind_left = arg_min # lower temp
        ind_right = arg_min -1
    t1, t2 = ref_temp[ind_left], ref_temp[ind_right]
    n1, n2 = ref_sptind[ind_left], ref_sptind[ind_right]
    num = (n2-n1)/(t2-t1)*(teff - t1) + n1
    return num


def convert_sptnum_to_temp(sptnum, option='Tpl', out_nan=False):
    key = spt_option_dic[option]

    ref_temp = spt_convert_table[key]
    ref_sptind = spt_convert_table['SpTind']
    roi_valid = np.isfinite(ref_temp)
    ref_temp = ref_temp[roi_valid]
    ref_sptind = ref_sptind[roi_valid]
    
    i_tmax = np.argmax(ref_temp)
    i_tmin = np.argmin(ref_temp)
    
    if sptnum < np.min(ref_sptind):
        if out_nan: return np.nan
        else: sys.exit("Out of the range: %s"%spt_desc_dic[option])
        # return ref_temp[i_tmax]
    if sptnum > np.max(ref_sptind):
        if out_nan: return np.nan
        else: sys.exit("Out of the range: %s"%spt_desc_dic[option])
        # return ref_temp[i_tmin]
    
    arg_min = np.argmin(abs(ref_sptind-sptnum))
    if ref_sptind[arg_min] >= sptnum:
        ind_right = arg_min # larger index 
        ind_left = arg_min -1 # sort from small to large index
    else:
        ind_left = arg_min # smaller index
        ind_right = arg_min +1
    t1, t2 = ref_temp[ind_left], ref_temp[ind_right]
    n1, n2 = ref_sptind[ind_left], ref_sptind[ind_right]
    temp = (t2-t1)/(n2-n1)*(sptnum - n1) + t1
    return temp

def convert_spt_to_temp(spt, option='Tpl', out_nan=False):
    return convert_sptnum_to_temp( convert_spt_to_num(spt), option=option, out_nan=out_nan )
def convert_temp_to_spt(temp, option='Tpl', out_nan=False):
    return convert_sptnum_to_spt( convert_temp_to_sptnum(temp, option=option, out_nan=out_nan))

# upgraded version
def make_spt(only_int=False, option='Tr14'):
    key = spt_option_dic[option]
    spt_avail = spt_convert_table['SpT'].data[np.isfinite(spt_convert_table[key])]
    if only_int:
        spt_array = np.array( [k for k in np.unique(spt_avail) if float(k[1:])==np.around(float(k[1:])) ] )
    else:
        spt_array = np.unique(spt_avail)

    spt_t_array = np.array([spt_convert_table[key][np.where(spt_convert_table['SpT']==k)[-1][0]] for k in spt_array])
    arg = np.argsort(spt_t_array)
    spt_t_array = spt_t_array[arg]
    spt_array = spt_array[arg]
    return spt_array, spt_t_array
    

# -----------------------------------------------------------------------------------

from .execute import get_posterior as get_post
from .execute import get_posterior_group as get_post_group

def get_posterior(y, astro, N=4096, unc=None, flag=None, return_llike=False, quiet=True, use_group=False, group=None):
    
    y_it = astro.obs_to_y(y)
    if astro.prenoise_training == True:
        if unc.shape == y.shape:
            sig = astro.unc_to_sig(unc)
            y_it = np.hstack((y_it, sig))
    
    if astro.use_flag == True:
        if flag.shape[-1] == len(astro.flag_names):
            if len(flag.shape)==1:
                flag2d = flag.reshape(1,-1)
                y2d = y_it.reshape(1,-1)
            else:
                flag2d = flag
                y2d = y_it

            for i_flag, flag_name in enumerate(astro.flag_names):
                roi_off = flag2d[:, i_flag] == 0.0
                y2d[roi_off][:,astro.flag_index_dic[flag_name]] = 0.0

            rf = astro.flag_to_rf(flag2d)
            y_it = np.hstack((y2d, rf))

    if astro.wavelength_coupling == True:
        wl = get_coupling_wavelength(astro.y_names)
        if len(y.shape)>1: 
            wl = np.repeat(wl.reshape(1,-1), y.shape[0], axis=0)
        y_it = np.hstack( (y_it, astro.wl_to_lambda(wl)))

            
    
    if use_group:
        output = get_post_group(y_it, astro, N=N, group=group, return_llike=return_llike, quiet=quiet)
    else:
        output = get_post(y_it, astro, N=N, return_llike=return_llike, quiet=quiet)
    if return_llike:
        x = astro.x_to_params(output[0])
        return x, output[1]
    else:
        return astro.x_to_params(output)
    
GPU_MAX_LOAD = 0.5         # Maximum compute load of GPU allowed in automated selection
GPU_MAX_MEMORY = 0.5         # Maximum memory load of GPU allowed in automated selection
GPU_WAIT_S = 600             # Time (s) to wait between tries to find a free GPU if none was found
GPU_ATTEMPTS = 10            # Number of times to retry finding a GPU if none was found
GPU_EXCLUDE_IDS = [] # List of GPU IDs that are to be ignored when trying to find a free GPU, leave as empty list if none

def find_gpu_available(gpu_max_load=GPU_MAX_LOAD, gpu_max_memory=GPU_MAX_MEMORY,
                       gpu_wait_s=GPU_WAIT_S, gpu_attempts=GPU_ATTEMPTS,
                       gpu_exclude_ids=GPU_EXCLUDE_IDS, verbose=True,
                       return_str=True, return_list=False):
    """
    Find and return available gpu. Only use this function if you have GPU

    Parameters
    ----------
    return_str : bool, optional
        get full str -> cuda:1  The default is True.
    return_list : bool, optional
        get full list of avaibable gpus [1,3,4]. The default is False.

    Returns
    -------
    str or list or int
        information of GPU available.

    """
    try:
        import GPUtil
        device_id_list = GPUtil.getFirstAvailable(maxLoad=gpu_max_load,
                                                  maxMemory=gpu_max_memory,
                                                  attempts=gpu_attempts,
                                                  interval=gpu_wait_s,
                                                  excludeID=gpu_exclude_ids,
                                                  verbose=verbose)
        if return_str:
            return 'cuda:{:d}'.format(device_id_list[0])
        elif return_list:
            return device_id_list
        else:
            return device_id_list[0]
    except Exception as e:
        print(e)
    
    

"""
functions for prenoise training: uncertainty calcualtion
"""

# create random uncertainty (dY/Y, sigma) in log scale for synthetic models
"""
size = tuple of (size[0], size[1], size[2],,,,)
correlation (correlation between different sigmas in one obs): 
    - Ind_Man : no corr between sigmas, all different sampling range (but same sampling method)
                HPs for sampling method are arrays
    - Ind_Unif : no corr between sigmas, the same sampling ftn for all sigmas
                HPs for sampling method are constants
                
sampling method: gaussian, uniform ([min, max), but exchange=> (min, max] )
"""    
def calculate_random_uncertainty(size, expand=1, correlation=None, sampling_method=None, y_names=None,
                       lsig_mean=None, lsig_std = None, lsig_min=None, lsig_max=None, 
                       flux=None, noise_pdf=None,  **kwarg ):
    # flux: whole spectrum. includ masked area. 
    
    if len(size)==1:
        oned = True
        size = (1, size[0])
    else:
        oned = False
        
    if expand > 1:
        size = (int(size[0]*expand), size[1] )
        
        
    if correlation == 'Ind_Unif': # all sig components are sampled from the same probability=p(sigma)
        if sampling_method == 'gaussian': # N(mean, std)  
            rn = np.random.randn(*size)*lsig_std + lsig_mean
        elif sampling_method == 'uniform': # [min, max)
            rn = np.random.rand(*size)*(lsig_max - lsig_min) + lsig_min
        lsig = rn
        
    elif correlation == 'Ind_Man': # every y component have different p(sigma) 
        lsig = np.zeros(size)
        if sampling_method == 'gaussian': # N(mean, std)  
            for i in range(lsig.shape[1]):
                lsig[:,i] = np.random.randn(lsig.shape[0])*lsig_std[i] + lsig_mean[i]
        elif sampling_method == 'uniform': # [min, max)
            for i in range(lsig.shape[1]):
                lsig[:,i] = np.random.rand(lsig.shape[0])*(lsig_max[i] - lsig_min[i]) + lsig_min[i]
    
    elif correlation == 'Single': # all y components have the same sigma value
        lsig = np.zeros(size)
        if sampling_method == 'gaussian': # N(mean, std)  
            rn = np.random.randn(lsig.shape[0])*lsig_std + lsig_mean
        elif sampling_method == 'uniform': # [min, max)
            rn = np.random.rand(lsig.shape[0])*(lsig_max - lsig_min) + lsig_min
        lsig += np.repeat(rn.reshape(-1,1), lsig.shape[1], axis=1)
        
    elif 'Seg_' in correlation: 
        # only for spectral y_components : 'l0','l20', etc,
        # use same sigma for each segment
        # you will get noise_pdf as kwarg if there is non spectral y component
        # this requires: y_names
        
        lsig = np.zeros(size)
        # 1) sample lsig for spectral components
        spec_indices = get_spec_index(y_names)
        spec_locs = get_spec_index(y_names, get_loc=True)
        wl = get_muse_wl()[spec_indices]
        
        iseg_list = divide_wl_segment(wl, **kwarg) # seg_size must be in kwarg, 
        # Don't need to set start and end -> automatically set
        
        if correlation == 'Seg_Flux': # sigma sampled follwoing the flux (Poisson+sth). only uniform available
            if sampling_method == 'uniform': # [min, max)
                lsig[:, spec_locs] = random_lsig_seg_flux_uniform(flux, iseg_list, lsig_min0=lsig_min, lsig_max0=lsig_max, **kwarg)
                # flux is alread in N_obs x N(spec_locs) size
                # lsig_min, max is only used for the minimum sigma value per obs
                
        elif correlation == 'Seg_Unif': # all sig componets are sampled from the same probability=p(sigma) but the same sigma within the segment
            for iseg, ind_seg in enumerate(iseg_list):
                if sampling_method=='uniform':
                    lsig[:, spec_locs[ind_seg]] = (np.random.rand(size[0]) * (lsig_max - lsig_min) + lsig_min).reshape(-1,1)
                elif sampling_method=='gaussian':
                    lsig[:, spec_locs[ind_seg]] = (np.random.randn(size[0]) * lsig_std + lsig_mean).reshape(-1,1)
                    
        # 2) sample lsig for non spectral components, if exist
        # you will get 'noise_pdf' kwarg, if non spectral component exists
        if noise_pdf is not None:
            for param, pdf in noise_pdf.items():
                # p(sigma) can be same as spectral components or not, it is already determined in noise_pdf
                i_param = y_names.index(param)
                if pdf['sampling']=='uniform':
                    lsig[:, i_param] = np.random.rand(size[0]) * (pdf['lsig_max'] - pdf['lsig_min']) + pdf['lsig_min']
                elif pdf['sampling']=='gaussian':
                    lsig[:, i_param] = np.random.randn(size[0]) * pdf['lsig_std'] + pdf['lsig_mean']

            
    # elif correlation =='Seg_Poisson': # random for only the min Sig (maxFlux), other seg are determined by Poisson
    #     # sampling method only determines how to sample the minSig
    #     iseg_list = divide_wl_segment(wl, **kwarg) # Don;t need to set start and end -> automatically set
    else:
        print('correlation is not set (Ind_Man, Ind_Unif, Single)')
        return None
    
    if oned==True and expand <=1:
        lsig = lsig.reshape(size[-1])
    
    return lsig

def random_lsig_seg_flux_uniform(flux, iseg_list, lsig_min0, lsig_max0, upper_limit=True, maxlgap=3, **kwarg):
    """
    Sampling random sigma based on flux of spectrum (segmented)    
    
    Parameters
    ----------
    flux : flux array including whole MUSE spectral bins. [n_obs x n_spec_bins]
    iseg_list : list of wl segement indices generated from divide_wl_segment
    lsig_min0 : min value of U(a,b) for the lowest sigma (highest flux segment)
    lsig_max0 : max value of U(a,b) for the lowest sigma (highest flux segment)
    upper_limit : T/F optional
        If True, highest value of log sig is limited to sig_lowest * 10^maxlgap. The default is True.
    maxlgap : set gatp optional
        limit highest value of log sig is limited to sig_lowest * 10^maxlgap. The default is 3.
        3 means sig_max is limited to lowest * 1e3

    Returns
    -------
    lsig : 2D array, the same shape as flux [n_obs x n_spec_bins]
        log sigma. sampled from uniform distribution. But p(sigma) differes in each segment

    """
    n_seg = len(iseg_list)
    flux = flux.reshape(-1, flux.shape[-1]) # make 2D flux array
    
    # Set order basedon median flux at each seg
    medflux_seg = np.zeros(shape=(flux.shape[0], n_seg))
    for iseg, ind_seg in enumerate(iseg_list):
        medflux_seg[:, iseg] = np.nanmedian(flux[:, ind_seg], axis=1)
    arg_flux = np.argsort(medflux_seg, axis=1)[:, ::-1] # 2D. n_obs x n_seg

    lsig = np.zeros(flux.shape)

    for i_obs, arg in enumerate(arg_flux):
        lsig_seg = np.zeros(len(arg))
        fseg = medflux_seg[i_obs]
        fseg_max = fseg[arg[0]] # arg[0] segment with the highest flux 
        for j_seg, j in enumerate(arg):
            if j_seg==0: # segment with the highest flux
                lsig_seg[j] = np.random.uniform(low=lsig_min0, high=lsig_max0, size=1)[0]
            else:
                low = lsig_seg[arg[j_seg-1]] # low(n) = previous lsig
                # high = lsig_seg[arg[j_seg-1]] + np.log10(np.sqrt(fseg_max/fseg[j])) # U( sig(n-1), sig(n-1) x Possion ratio -> as maximum
                high = lsig_seg[arg[0]] + np.log10(np.sqrt(fseg_max/fseg[j]) )  # U( sig(n-1), sig(n-1) x Possion ratio  x2-> as maximum
                if upper_limit:
                    # avoid large gap between sig_min and sig_max
                    # for example, lowest + 3 = lsig_seg[arg[0]] + 3 (maxlgap) # largest sigma is not bigger than 1000xlowsest sigma
                    high=np.min([high, lsig_seg[arg[0]] + maxlgap])
                lsig_seg[j] = np.random.uniform(low=low, high=high, size=1)[0]
            lsig[i_obs][iseg_list[j]] = lsig_seg[j]

    return lsig


def divide_wl_segment(wl, seg_size=500, start=1000, end=12000,  **kwarg):
    """
    make wavelength segment for uncertainty sampling
    return list of indices

    Parameters
    ----------
    wl : arry of wavelength (give whole spectral range)
        wavelength of spectrum used as input to the network
    start : value. treated minimum wavelength. for optical it is 1000A
        The default is 1000. You can set to cut the wavelength but in general it is not needed
    end : value. treated maximum wavelength., optional
        The default is 12000.
    seg_size : segment size. 200A, 500A , 1000A, the same unit as wl
        The default is 500 (AA)

    Returns
    -------
    iseg_list : list of arrays that contatining indices for each segment
        

    """
    wl_min = np.min(wl); wl_max = np.max(wl)
    if wl_min < start: 
        sys.exit("min(wl) <  %f"%start)
    else:
        start = np.floor(wl_min/seg_size)*seg_size
    if wl_max > end: 
        sys.exit("max(wl) > %f"%end)
    else:
        end = np.ceil(wl_max/seg_size)*seg_size
    
    n_seg = int((end-start)/seg_size)
    iseg_list = []
    for i_seg in range(n_seg):
        iseg_list.append( np.where( (wl >= start+seg_size*i_seg)*(wl < start+seg_size*(i_seg+1)) )[0] )
    return iseg_list
  
     

        
############################################################################
title_unit_dic = {
    'logTeff': r"log T$_{\mathrm{eff}}$ (K)",
    'Teff': r'T$_{\mathrm{eff}}$ (K)',
    'logG': r'log g (cm s$^{-2}$)',
    'A_V': r'A$_{\mathrm{V}}$ (mag)',
    'veil_r': r'r$_{\mathrm{veil}}$',
    'library': 'Library',
    "R_V": r'R$_{\mathrm{V}}$',
    
    
    # slab parameters
    'Tslab': r"T$_{\mathrm{slab}}$",
    'log_ne': r"log n$_{\mathrm{e,slab}}$ (cm$^{-3}$)",
    'ne': r"n$_{\mathrm{e,slab}}$ (cm$^{-3}$)",
    
    'log_tau0': r"log $\tau_{\mathrm{0,slab}}$",
    'tau0': r"$\tau_{\mathrm{0,slab}}$",
    'log_Fslab':r"log F$_{\mathrm{slab,norm}}$",
}

title_dic = {
    'logTeff': r"log T$_{\mathrm{eff}}$",
    'Teff': r'T$_{\mathrm{eff}}$',
    'logG': 'log g',
    'A_V': r'A$_{\mathrm{V}}$',
    'library': 'Library',
    'veil_r': r'r$_{\mathrm{veil}}$',
    "R_V": r'R$_{\mathrm{V}}$',
    
    # slab parameters
    'Tslab': r"T$_{\mathrm{slab}}$",
    'log_ne': r"log n$_{\mathrm{e,slab}}$",
    'ne': r"n$_{\mathrm{e,slab}}$",
    'log_tau0': r"log $\tau_{\mathrm{0,slab}}$",
    'tau0': r"$\tau_{\mathrm{0,slab}}$",
    'log_Fslab':r"log F$_{\mathrm{slab,norm}}$",
    
    
    
    'tT': r'T$_{\mathrm{eff}}^{\mathrm{True}}$',
    'mT': r'T$_{\mathrm{eff}}^{\mathrm{MAP}}$',
    'dT': r'$\Delta$ T$_{\mathrm{eff}}$',
    'tG': r'log g$^{\mathrm{True}}$',
    'mG': r'log g$^{\mathrm{MAP}}$',
    'dG': r'$\Delta$ log g',
    'tA': r'A$_{\mathrm{V}}^{\mathrm{True}}$',
    'mA': r'A$_{\mathrm{V}}^{\mathrm{MAP}}$',
    'dA': r'$\Delta$ A$_{\mathrm{V}}$',
} 


def get_title(key, unit=True):
    if unit:
        try:
            return title_unit_dic[key]
        except:
            return key
    else:
        try:
            return title_dic[key]
        except:
            return key

    
"""
Useful functions for analysis
"""
def calculate_uncertainty(parameter_distr, astro, confidence=68, percent=True,
                          add_Teff=False, spt_scale=None, return_percentile=False,
                          ):
                          # exclude_infinite=True, exclude_unphysical=False, **kwarg):
    
    parameter_distr = parameter_distr.copy()
    npost = len(parameter_distr)
    
    # if exclude_unphysical:
    #     roi_physical = exclude_models(parameter_distr, astro, exclude_infinite=exclude_infinite,
    #                                 exclude_unphysical=exclude_unphysical,
    #                                   do_extrp=False, return_dic=False, **kwarg) 
    #     parameter_distr = parameter_distr[roi_physical]
    # elif exclude_infinite:
    #     roi_finite = exclude_models(parameter_distr, astro, exclude_infinite=exclude_infinite,
    #                                 exclude_unphysical=exclude_unphysical,
    #                                   do_extrp=False, return_dic=False, **kwarg)
    #     parameter_distr = parameter_distr[roi_finite]
        
        
    # check finite values (recommend to make default but to avoid redundant checking)
    if np.sum(np.isfinite(parameter_distr)==False)>0: # only there is any infinite value
        roi_finite = np.array([True]*len(parameter_distr))
        for i_col in range(parameter_distr.shape[1]):
            roi_finite *= np.isfinite(parameter_distr[:,i_col])
        parameter_distr = parameter_distr[roi_finite]
        
        
    if percent:
        confidence *= 1e-2
    
    if add_Teff:
        if 'logTeff' not in astro.x_names:
            add_Teff = False

    unc_list = []
    per_list = []
    if len(parameter_distr) > 0.1*npost:
        q_low  = 100. * 0.5 * (1 - confidence)
        q_high = 100. * 0.5 * (1 + confidence)    
        
        for i, param in enumerate(astro.x_names): 
            x_low, x_high = np.nanpercentile(parameter_distr[:, i], [q_low, q_high])
            unc_list.append(x_high - x_low)
            per_list.append([x_low, x_high])
            
        if add_Teff:
            i_logTeff = astro.x_names.index('logTeff')
            x_low, x_high = np.nanpercentile(parameter_distr[:, i_logTeff], [q_low, q_high])
            unc_list.append( 10**x_high - 10** x_low)
            per_list.append([10** x_low, 10**x_high])
    else:
        unc_list = [np.nan]*len(astro.x_names)
        per_list = [[np.nan,np.nan]]*len(astro.x_names)
        if add_Teff:
            unc_list.append(np.nan)
            per_list.append([[np.nan,np.nan]])
    
    unc_list = np.array(unc_list)
    per_list = np.array(per_list)
            
    if return_percentile:
        return (unc_list, per_list)
        
    else:
        return unc_list


# calculate_map, if plot=True, give simple horizontal figure
def calculate_map(parameter_distr, astro, 
                  bw_method = 'silverman', n_grid=1024, use_percentile = None,
                  exclude_unphysical=False, 
                  plot=False,  return_figure=False, plot_model = True, plot_map = True, plot_kde = True,
                  nrow = None, ncol=None, figsize=[10,7],
                  xranges_dic=None, nbin=100,  model_distr=None, bar_plot = True,
                  color_post='gray', alpha=0.4, color_kde=None,color_model='red',color_map = 'orange',
                  legend=True,legend_post='posterior', legend_model=r'X$^{\mathrm{True}}$', legend_map=r'X$^{\mathrm{MAP}}$',
                  ind_legend=0,
                  ylabelsize='medium',xlabelsize='large',legendsize='small',ticklabelsize='medium', **kwarg):

    from KDEpy import FFTKDE
    
    parameter_distr = parameter_distr.copy()
    npost = len(parameter_distr)
    
    # check finite values (recommend to make default but to avoid redundant checking)
    if np.sum(np.isfinite(parameter_distr)==False)>0: # only there is any infinite value
        roi_finite = np.array([True]*len(parameter_distr))
        for i_col in range(parameter_distr.shape[1]):
            roi_finite *= np.isfinite(parameter_distr[:,i_col])
        parameter_distr = parameter_distr[roi_finite]
    
    # if exclude_unphysical:
    #     roi_physical = exclude_models(parameter_distr, astro, exclude_infinite=exclude_infinite, exclude_unphysical=exclude_unphysical,
    #                                   do_extrp=False, return_dic=False, **kwarg) 
    #     # **kwarg should contain: exclude_age_agey=T/F, exclude_age_agey_n1=T/F, print_log=T/F, agey_thre=5e5
    #     parameter_distr = parameter_distr[roi_physical]
        
    

    # Check plot condition in advance if plot=True
    if plot:
        # true value info? 
        if model_distr is not None:
            model_distr = model_distr.copy()
        else:
            plot_model = False
            
        # xrange set?
        if xranges_dic is not None:
            check_range = True
        else:
            check_range = False
            xranges_dic = {}
            
        # plot setting
        if not (nrow!=None and ncol!=None):
            nrow = np.ceil(len(astro.x_names)/4).astype(int)
            ncol = np.ceil(len(astro.x_names)/nrow).astype(int)
            figsize=[2.5*ncol, 3.5*nrow]
            
        fig, axis = plt.subplots(nrow, ncol, figsize=figsize)
        axis = axis.ravel()
        
        
    plot_index = 0

    map_list = np.zeros(len(astro.x_names))
    
    for i_param, param in enumerate(astro.x_names):

        post = parameter_distr[:, i_param].copy() # 1D
        
        # preprocess any parameter? e.g. age..., log.. 
        # process both post and true and change title
        title = param
        
        # KDE
        # roi_kde = np.array([True]*len(parameter_distr))
        roi_kde = np.isfinite(post)
        
        if use_percentile is not None:
            if (use_percentile > 0)*(use_percentile <100):
                q_low  = 0.5 * (100 - use_percentile)
                q_high = 0.5 * (100 + use_percentile)
                x_low, x_high = np.nanpercentile(post, [q_low, q_high] )
                roi_kde = (post>=x_low)*(post<=x_high)
        
        # kde: only run when kde posteriors > 10% of input posteirors
        if np.sum(roi_kde) > 0.1*npost:
            try:
                kde = FFTKDE(kernel='gaussian', bw=bw_method).fit(post[roi_kde])
                
                pmin, pmax = np.nanmin(post[roi_kde]), np.nanmax(post[roi_kde])
                x_grid = np.linspace( pmin - (pmax-pmin)/(n_grid*2), pmax + (pmax-pmin)/(n_grid*2), n_grid)
                y_grid = kde.evaluate(x_grid)
                x_map = x_grid[np.argmax(y_grid)]
            except: # Exception as e:
                x_map = np.nan
                # print("[MAP error] )
        
        else:
            x_map = np.nan
            
        map_list[i_param] = x_map
            
        # plot
        if plot:
            if plot_model:
                true_val = model_distr[i_param]
            
            if check_range:
                if param in xranges_dic:
                    xrange = xranges_dic[param]
                else:
                    xrange = None
            else:
                xrange = None

            bins = nbin
            labels = None
            
            
            # 2. calculate histogram
            yhis, _xhis = np.histogram(post, bins=bins, density=True, range=xrange)
            xhis = 0.5*(_xhis[:-1]+_xhis[1:])

            ax = axis[plot_index]
            if labels:
                ax.bar(np.arange(len(yhis))+1, yhis , color=color_post)
                if plot_model:
                    gt, _ = np.histogram(np.zeros(2)+true_val, bins=_xhis, density=True)
                    gt = yhis * gt
                    ax.bar(np.arange(len(gt))+1, gt, color='r', fill=False, linewidth=2.5, edgecolor='r')
                ax.set_xticks(np.arange(len(yhis))+1)
                ax.set_xticklabels(labels)
            else: 
                ax.step(xhis, yhis, where='mid', color=color_post)#,label='posterior')
                l1=ax.fill_between(xhis, yhis, ec=color_post, fc=color_post, alpha=alpha, label=legend_post)
                if xrange is not None:
                    ax.set_xlim(xrange)
                if plot_model:
                    ax.axvline(x=true_val, color='r', ls='-', label=legend_model)
                if plot_kde:
                    ax.plot(x_grid, y_grid, color=color_kde, lw=1)
                if plot_map:   
                    ax.axvline(x=x_map, color=color_map, ls='--', label=legend_map)
                if legend:
                    if i_param==ind_legend:
                        ax.legend(loc='best',fontsize=legendsize)

            ax.set_xlabel('%s' %title, fontsize=xlabelsize)
            ax.set_ylabel('Probability density', fontsize=ylabelsize)
            ax.tick_params(axis='both',labelsize=ticklabelsize)      

        plot_index+=1

    if plot:
        for i in np.arange(plot_index, len(axis), 1):
            axis[i].axis("off")
        fig.tight_layout()
        if return_figure:
            return map_list, fig, axis
    
    return map_list



    
"""
Useful functions for plotting
"""

import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy

def plot_posterior(posterior, axis, c, x_names = None,
                   x_true=None, plot_true=True, nbin=100, xranges_dic=None, return_xranges_dic=False,
                    color_post = 'gray', alpha=0.4, text_true=True,
                     map_values=None, plot_map=True, color_map='orange', text_map=True, 
                     u68_values = None, calculate_u68=True,  text_u68=True, additional_text=None,
                     ylabelsize='medium', ylabel='Probability density', xlabelsize='large', 
                     yticklabelsize='medium',xticklabelsize='large',
                     txtsize='small', title_unit=True,
                      **kwarg):
    """
    Basic plot posterior

    Parameters
    ----------
    posterior : NxM ndarray 
        NxM, N: number of posterior estimates, M: number of parameters
    axis : matplotlib axis
    c : config class
    x_names: if you want to only use some of the parmaeters. use this.
    Returns
    -------
    None.

    """
    if x_names is not None:
        # check dimension 
        if posterior.shape[1] != len(x_names):
            raise ValueError("posterior and x_names dimension mismatch")
        param_names = x_names
    else:
        param_names = c.x_names.copy()
        
    n_param = len(param_names)
    
    # check dimension 
    if posterior.shape[1] != n_param:
        raise ValueError("posterior and parameter dimension mismatch")
    
    
    # Check availabily for options
    x_true_given = False
    if x_true is not None:
        if len(x_true)==n_param:
            x_true_given = True
    plot_true = plot_true*x_true_given
    text_true = text_true*x_true_given
        
    map_values_given = False
    if map_values is not None:
        if len(map_values)==n_param:
            map_values_given = True
    plot_map = plot_map*map_values_given
    text_map = text_map*map_values_given
  
    u68_given = False
    if u68_values is not None:
        if len(u68_values)==n_param:
            u68_given = True
        elif len(u68_values)==(n_param+1) and 'logTeff' in param_names:
            u68_given = True
    if (u68_given is False):
        if calculate_u68*text_u68:
            u68_values = calculate_uncertainty(posterior, c, confidence=68, percent=True,
                          add_Teff=bool('logTeff' in param_names) )
            u68_given = True
            
        else:
            text_u68 = False
            
    if return_xranges_dic==True and xranges_dic == None:
        xranges_dic = {}
    

    for i_param, param in enumerate(param_names):
        
        post = posterior[:, i_param].copy() # 1D
        if x_true_given:
            true_val = x_true[i_param]
        if map_values_given:
            x_map = map_values[i_param]
        if u68_given:
            u68 = u68_values[i_param]
        
        if title_unit:
            try:
                title = title_unit_dic[param]
            except:
                title = param
        else:
            try:
                title = title_dic[param]
            except:
                title = param
              
        # Histogram plot
        bins = nbin
        labels = None
        
        # 2. calculate histogram
        xrange = None
        if xranges_dic is not None:
            if param in xranges_dic:
                xrange = xranges_dic[param]

        yhis, _xhis = np.histogram(post, bins=bins, density=True, range=xrange)
        xhis = 0.5*(_xhis[:-1]+_xhis[1:])
        if return_xranges_dic:
            xranges_dic[param] = [min(_xhis), max(_xhis)]
                       
        
        ax = axis[i_param]
        if labels:
            ax.bar(np.arange(len(yhis))+1, yhis , label='posterior', color=color_post)
            if plot_true:
                if np.isfinite(true_val):
                    gt, _ = np.histogram(np.zeros(2)+true_val, bins=_xhis, density=True)
                    gt = yhis * gt
                    ax.bar(np.arange(len(gt))+1, gt, color='r', fill=False, linewidth=2.5, edgecolor='r')
            ax.set_xticks(np.arange(len(yhis))+1)
            ax.set_xticklabels(labels)
  
        else:
            ax.step(xhis, yhis, where='mid', color=color_post)#,label='posterior')
            ax.fill_between(xhis, yhis, color=color_post, alpha=alpha)#, label='posterior') 
            
            if xrange is not None:
                xlim = ax.get_xlim()
                
            if plot_true:
                if np.isfinite(true_val):
                    ax.axvline(x=true_val, color='r', ls='-', label=r'$X^{\mathrm{True}}$=%#.4g'%(true_val))
            if plot_map:
                ax.axvline(x=x_map, color=color_map, ls='--',)
            
            if xrange is not None:
                ax.set_xlim(xlim)
    
            txt = []
            if text_true:
                if np.isfinite(true_val):
                    txt.append(r'$X^{\mathrm{True}}$=%#.4g'%(true_val))
                    if param=='logTeff':
                        txt.append('(%.5g [K])'%10**x_true[i_param])
            if text_map:
                txt.append( r'$X^{\mathrm{MAP}}$=%#.4g'%(x_map)     )
                if param=='logTeff':
                    txt.append('(%.5g [K])'%10**x_map)
              
            if text_u68:
                txt.append('$u_{68}$=%#.4g'%(u68))
                if param=='logTeff' and len(u68_values)==(n_param+1):
                    txt.append('(%.5g [K])'%( u68_values[-1]))
                    
                    
            if additional_text is not None:
                txt.append(additional_text)
        
            if len(txt)>0:
                ax.text(0.05,0.95, '\n'.join(txt), ha='left',  transform= ax.transAxes, va='top' , fontsize=txtsize,
                            bbox=dict(boxstyle='round', facecolor='w', alpha=0.6, edgecolor='silver') )
            
      
            ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
        

        ax.set_xlabel(title, fontsize=xlabelsize)
        ax.set_ylabel(ylabel, fontsize=ylabelsize)
        ax.tick_params(axis='y',which='major',labelsize=yticklabelsize)
        ax.tick_params(axis='x',which='major',labelsize=xticklabelsize)
    
    if return_xranges_dic:
        return xranges_dic
        
        
def overplot_posterior(posterior, axis, c, x_names = None,
                       nbin=100, hratio=0.5, ls=':', color_post='C0', alpha=0.4,
                       xranges_dic=None,legend_label=None, plot_legend=True, legend_size='small', **kwarg):
    
    if x_names is not None:
        # check dimension 
        if posterior.shape[1] != len(x_names):
            raise ValueError("posterior and x_names dimension mismatch")
        param_names = x_names
    else:
        param_names = c.x_names.copy()
        
    n_param = len(param_names)
    
    # check dimension 
    if posterior.shape[1] != n_param:
        raise ValueError("posterior and parameter dimension mismatch")

    for i_param, param in enumerate(param_names):
      
      post = posterior[:, i_param].copy() # 1D
            
      # Histogram plot
      bins = nbin
      labels = None
      
      # 2. calculate histogram
      xrange = None
      if xranges_dic is not None:
          if param in xranges_dic:
              xrange = xranges_dic[param]

      yhis, _xhis = np.histogram(post, bins=bins, density=True, range=xrange)
      xhis = 0.5*(_xhis[:-1]+_xhis[1:])
      yhis = yhis * hratio
      
      ax = axis[i_param]
      # return previous axis
      xr = ax.get_xlim()
      yr = ax.get_ylim()
      
      if labels:
          ax.bar(np.arange(len(yhis))+1, yhis , label=legend_label, color=color_post)
          
          ax.set_xticks(np.arange(len(yhis))+1)
          ax.set_xticklabels(labels)

      else:
          ax.step(xhis, yhis, where='mid', color=color_post, ls=ls)
          ax.fill_between(xhis, yhis, color=color_post, alpha=alpha, label=legend_label)
          
      ax.set_xlim(xr)
      ax.set_ylim(yr)
          
          
      if i_param==0 and plot_legend==True:
          # if ax.get_legend():
          ax.legend(loc='best', fontsize=legend_size)
          
    
    

# plot useful scatter plots
def set_basic(ax, xrange=None, yrange=None,xlabel=None, ylabel=None, title=None, 
              xlabelsize='large', ylabelsize='large', titlesize='x-large',
              xticklabelsize='medium', yticklabelsize='medium', 
             **kwarg,):
    
    if xrange: ax.set_xlim(xrange)
    if yrange: ax.set_ylim(yrange)
    if title: ax.set_title(title, size=titlesize)
    if xlabel: ax.set_xlabel(xlabel, size=xlabelsize)
    if ylabel: ax.set_ylabel(ylabel, size=ylabelsize)
    
    ax.tick_params(axis='x', which='major', labelsize=xticklabelsize)
    ax.tick_params(axis='y', which='major', labelsize=yticklabelsize)
    

def scatter_color(fig, ax, xval, yval, cval, sval=40, plot_cbar=True, 
                    marker='o', edgecolors='grey', cmap = plt.get_cmap("RdYlBu"), vmin=None, vmax=None,          
                    xrange=None, yrange=None, alpha=1,
                    xlabel=None, ylabel=None, title=None, label=None,
                    xlabelsize='large', ylabelsize='large', titlesize='x-large',
                    xticklabelsize='medium', yticklabelsize='medium', txtsize = 'small',
                    cbar_position='right', cbar_size="5%", cbar_pad="1%", cbar_ticklabelsize='small',
                    cbar_title=None, cbar_titlesize='small', cbar_Ntick=3, cbar_title_rotation=None,
                    size_factor=15, legend_size=True, legend_color_value=0.8, legend_position='best',
                    legend_titlesize='small', legend_title=None, legend_alpha=0.7, legend_Nsize=5,
                    return_ims = False,
                   **kwarg,
   
                 ):
    
    if vmin is None:
        vmin = np.nanmin(cval)
    if vmax is None:
        vmax = np.nanmax(cval)
        
    try:
        a=len(sval)
        sval = sval * size_factor
    except:
        legend_size = False
    
    ims=ax.scatter(xval, yval, marker=marker, c=cval, s=sval, cmap=cmap, label=label,
                   vmin=vmin, vmax=vmax, edgecolors=edgecolors, alpha=alpha)
    
    set_basic(ax,  xrange=xrange, yrange=yrange, xlabel=xlabel, ylabel=ylabel, title=title, 
              xlabelsize=xlabelsize, ylabelsize=ylabelsize, titlesize=titlesize,
              xticklabelsize=xticklabelsize, yticklabelsize=yticklabelsize)
    
    if plot_cbar:
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes(cbar_position, size=cbar_size, pad=cbar_pad)
        # (left, right, bottom, top)
        tick_on = np.array([False, False, False, False])
        orientation = 'vertical'
        if cbar_position=='right':
            tick_on[1]=True
            rotation = 270
        elif cbar_position=='left':
            tick_on[0] = True
            rotation = 90
        elif cbar_position =='bottom':
            tick_on[2] = True
            rotation = 0
            orientation='horizontal'
        elif cbar_position =='top':
            tick_on[3] = True
            orientation='horizontal'
            rotation = 0
        if cbar_title_rotation is not None:
            rotation = cbar_title_rotation
            
        cbar = fig.colorbar(ims, cax=cbar_ax, orientation=orientation)
        cbar_ax.tick_params(axis='both', which='both', labelsize=cbar_ticklabelsize, 
                            left = tick_on[0], right=tick_on[1], bottom=tick_on[2], top=tick_on[3],
                            labelleft = tick_on[0], labelright=tick_on[1], labelbottom=tick_on[2], labeltop=tick_on[3] )
        if orientation=='vertical':
            cbar_ax.yaxis.set_major_locator(ticker.MaxNLocator(cbar_Ntick))
            cbar_ax.set_ylabel(cbar_title, size=cbar_titlesize)
        else:
            cbar_ax.xaxis.set_major_locator(ticker.MaxNLocator(cbar_Ntick))
            cbar_ax.set_xlabel(cbar_title, size=cbar_titlesize)
        # cbar.set_label(cbar_title, size=cbar_titlesize, rotation=rotation)
        
    if legend_size:
        kw = dict(prop="sizes", num=legend_Nsize, color=ims.cmap(legend_color_value), fmt="{x:.2f}",
          func=lambda s: s/size_factor)
        legend2 = ax.legend(*ims.legend_elements(**kw), framealpha=legend_alpha,
                            loc=legend_position, title=legend_title, fontsize=legend_titlesize)

    if return_ims:
        return ims
