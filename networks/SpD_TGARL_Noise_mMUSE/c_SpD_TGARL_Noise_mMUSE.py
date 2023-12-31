"""
original SpD_TGARL_Noise_MUSE_ntf_31 in hpsb1
Settl + Dusty library
Noise-Net
T: 2600~7000
g: 2.5 ~ 5
Av: 0 ~ 10 (random sampling during training) + f_min = 0.1
r: 0 ~ 2 (random sampling during tarining)  + f_min = 0.5

library : 0=Settl, 1=Dusty
use the entire MUSE range but exclude masked regions (emission lines)
"""
 
# x_names and y_names according to DB (tablename) column names
x_names = [
	'logTeff' ,
	'logG' ,
	'A_V' ,
	'veil_r' ,
	'library'
]
y_names = ['l{:d}'.format(k) for k in range(0, 82)]+['l{:d}'.format(k) for k in range(95, 164)]+['l{:d}'.format(k) for k in range(171, 201)]+['l{:d}'.format(k) for k in range(210, 660)]+['l{:d}'.format(k) for k in range(666, 897)]+['l{:d}'.format(k) for k in range(906, 1237)]+['l{:d}'.format(k) for k in range(1243, 1435)]+['l{:d}'.format(k) for k in range(1442, 1442)]+['l{:d}'.format(k) for k in range(1471, 1538)]+['l{:d}'.format(k) for k in range(1545, 1568)]+['l{:d}'.format(k) for k in range(1578, 1583)]+['l{:d}'.format(k) for k in range(1589, 1851)]+['l{:d}'.format(k) for k in range(1854, 1904)]+['l{:d}'.format(k) for k in range(1913, 2023)]+['l{:d}'.format(k) for k in range(2027, 2053)]+['l{:d}'.format(k) for k in range(2059, 2061)]+['l{:d}'.format(k) for k in range(2066, 2396)]+['l{:d}'.format(k) for k in range(2403, 2414)]+['l{:d}'.format(k) for k in range(2425, 2953)]+['l{:d}'.format(k) for k in range(2963, 2996)]+['l{:d}'.format(k) for k in range(3004, 3029)]+['l{:d}'.format(k) for k in range(3042, 3077)]+['l{:d}'.format(k) for k in range(3085, 3127)]+['l{:d}'.format(k) for k in range(3137, 3197)]+['l{:d}'.format(k) for k in range(3207, 3286)]+['l{:d}'.format(k) for k in range(3294, 3407)]+['l{:d}'.format(k) for k in range(3414, 3451)]+['l{:d}'.format(k) for k in range(3459, 3577)]+['l{:d}'.format(k) for k in range(3589, 3681)]
 
# These parameters will be randomly selected during the training from corresponding ranges
random_parameters = {'A_V': (0, 10), 'veil_r': (0, 2)}
 
# These are kwarg needed in some functions like extinction 
additional_kwarg = {'R_V': 4.4, 'f_min_dic': {'A_V': 0.1, 'veil_r': 0.3}}
 
# name of output network, Ex) output/xxxx_network.pt
# Do not forget to make upper directory if you use
filename = 'SpD_TGARL_Noise_mMUSE.pt'
 
# name (inculuding path) of database
tablename = 'Database/ecogal_spectra_ts_Settl_plus_Dusty_Pristine.csv'
 
# if you use network outside your proj_dir
# you can set proj_dir by > read_config_from_file('path_to_configfile',  proj_dir = proj_dir)
# This automatically updates filename and tablename: proj_dir+filename
# you can control with keywords:
# change_filename=True/False, change_tablename=True/False
 
# Set model_code:'ModelAdamGLOW' , 'ModelAdamAllInOne'
architecture = 'cINN'
model_code = 'ModelAdamGLOW'
FrEIA_ver = 0.2
 
# fraction of test set: 0 < test_frac < 1
test_frac = 0.2
 
# smoothing discretized parameters (adding Gaussian noise)
train_smoothing = True
smoothing_sigma = {'library': 0.05}
 
# Normalize spectra (option: normalize_total_flux, normalize_mean_flux)
normalize_flux = True
normalize_total_flux = True
normalize_mean_flux = None
normalize_f750 = None
 
# Prenoise training (Soft training): Use dY/Y as additional conditions
prenoise_training = True
n_sig_MC = 1 # This must be 1 (will be deprecated)
n_noise_MC = 1 # This must be 1 (will be deprecated)
# Correlation between uncertainies in one obs
unc_corrl = 'Single'
# Distribution of uncertainty sampling
unc_sampling = 'uniform'
lsig_min = -5
lsig_max = -0.5
 
# Network with y flag : control turning on/off of certain y components
use_flag = False
"""
 Training hyperparameters 
"""
device = 'cuda'
# device: cpu, cuda, cuda:0 (modifiable anytime after you read config)
batch_size = 512
n_epochs = 177
 
pre_low_lr = 0
n_its_per_epoch = 2**16
 
load_file = ''
 
checkpoint_save = False
checkpoint_save_interval = 240
checkpoint_save_overwrite = True
 
"""
 Learning Optimization 
"""
gamma = 0.41546315358922503
lr_init = 0.0002100524764996379
l2_weight_reg = 8.217188498056922e-05
adam_betas = (0.8, 0.8)
meta_epoch = 12
# how often you change the learning rate (1=every epch)
 
seed_weight_init = 141625
# Seed for the random network weight initialisation. The default is 1234.
 
do_rev = False
 
"""
 Model Construction 
"""
n_blocks = 8
 
# subnetwork's internal width, the number of layers
internal_width = 512
internal_layer = 3
init_scale = 0.03
# feature_net (cond_net): (y is converted to feature through cond_net to be used as conditions)
y_dim_features = 256
# dimension used in cond_net 
feature_width = 256
feature_layer = 4
# cond_net's width and layers
 
# ModelAdamGLOW hyperparameters (2)
exponent_clamping = 2.0
use_permutation = True
 
"""
 Visualization 
"""
loss_names = ['Loss_train_mn', 'Loss_test_mn', 'Loss_train_mdn', 'Loss_test_mdn']
loss_plot_xrange = None
loss_plot_yrange = [-40, 10]
progress_bar = False
live_visualization = False
 
