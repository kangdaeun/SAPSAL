"""
v3 UV-Net.
original version: FpSpD_TGARsL_Noise_EfXS.pt in  test_xs_wg00

"""
 
# x_names and y_names according to DB (tablename) column names
x_names =     [ 'logTeff', 'logG', 'A_V' , 'log_veil_r', 'library',
	'Tslab', 'log_ne', 'log_tau0', 'log_Fslab']
 
y_names =   [
	'R_V', 'f3400', 'f3550', 'f3605', 'f4005', 'f4145', 'f4650', 'f4750', 'f5125', 'f5415',
	'f6255', 'f6447.5', 'f6630', 'f6825', 'f7030', 'f7070', 'f7100', 'f7140', 'f7200', 'f7400',
	'f7500', 'f7560', 'f7975','f8100', 'f8575', 'f8630', 'f8710'
]
 
# These parameters will be randomly selected during the training from corresponding ranges
random_parameters = {'A_V': (0, 10), 'log_veil_r': (-4, 1), 'R_V': (2, 6)}
 
# These are kwarg needed in some functions like extinction 
additional_kwarg = {'f_min_dic': {'A_V': 0.1}, 'R_V_noise_pdf': {'sampling': 'uniform', 'lsig_min': -5, 'lsig_max': -0.5}, 'wl_in_str': 'np.concatenate( [np.arange(3200, np.floor(4750.1572265625), 1), np.arange(4750.1572265625, 9351.4072265625, 1.25) ] )'}
 
 
# name of output network, Ex) output/xxxx_network.pt
# Do not forget to make upper directory if you use
filename = 'v3_UV_Net.pt'
 
# name (inculuding path) of database
tablename = 'Database/SpDpFRP_XSLowRes_2p18.parquet'
 
# if you use network outside your proj_dir
# you can set proj_dir by > read_config_from_file('path_to_configfile',  proj_dir = proj_dir)
# This automatically updates filename and tablename: proj_dir+filename
# you can control with keywords:
# change_filename=True/False, change_tablename=True/False
 
# Set model_code:'ModelAdamGLOW' , 'ModelAdamAllInOne'
model_code = 'ModelAdamGLOW'
 
# Set cond_net_code:'linear' , 'hybrid_cnn', 'hybrid_stack'. if not set, default is linear
cond_net_code = 'hybrid_stack'
conv_net_config = {'start_channels': 16, 'kernel_size_filter': 3, 'kernel_size_pooling': 2, 'stride_filter': 1, 'stride_pooling': 1, 'in_dim_conv': 26, 'in_channels': 2}
global_net_config = {'out_dim_global': 8, 'n_layers_global': 3, 'in_dim_global': 2}
 
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
 
# Veiling option (using Hydrogen Slab model for veiling)
use_Hslab_veiling = True
use_one_Hslab_model = None # If True, it use one example slab model for all. slab_grid is ignored
slab_grid = 'Database/Hslab_tau750_p15_XSLowRes.parquet' # Path to slab grid file (.csv)
# If use_Hslab_veiling=True but use_one_Hslab_model !=True, then automatically read slab_grid
 
# Prenoise training (Noise-Net): Use dY/Y as additional conditions
prenoise_training = True
# Correlation between uncertainies in one obs
unc_corrl = 'Seg_Unif'
# Distribution of uncertainty sampling
unc_sampling = 'uniform'
lsig_min = -5
lsig_max = 0
wl_seg_size = 500
 
# Network with y flag : control turning on/off of certain y components
use_flag = False
 
# Network with wavelength coupling
wavelength_coupling = False
 
# Domain Adaptaion: Use real data to improve simulation gap
domain_adaptation = False
 
"""
 Training hyperparameters 
"""
device = 'cpu'
# device: cpu, cuda, cuda:0 (modifiable anytime after you read config)
batch_size = 512
n_epochs = 211
 
pre_low_lr = 0
n_its_per_epoch = 2**16
 
load_file = ''
 
checkpoint_save = True
checkpoint_save_interval = 20
checkpoint_save_overwrite = True
checkpoint_remove_after_training = True
 
"""
 Learning Optimization 
"""
gamma = 0.6512701567216194
lr_init = 0.0009002029546913083
l2_weight_reg = 0.0001047536499350801
adam_betas = (0.5, 0.9)
meta_epoch = 11
# how often you change the learning rate (1=every epch)
 
seed_weight_init = 69593
# Seed for the random network weight initialisation. The default is 1234.
 
do_rev = False
 
"""
 Model Construction 
"""
n_blocks = 14
 
# subnetwork's internal width, the number of layers
internal_width = 256
internal_layer = 4
init_scale = 0.03
# feature_net (cond_net): (y is converted to feature through cond_net to be used as conditions)
y_dim_features = 128
# dimension used in cond_net 
feature_width = 512
feature_layer = 6
# cond_net's width and layers
 
# ModelAdamGLOW hyperparameters (2)
exponent_clamping = 2.0
use_permutation = True
 
"""
 Visualization 
"""
loss_names = ['Loss_train_mdn', 'Loss_test_mdn', 'Loss_train_mn', 'Loss_test_mn']
loss_plot_xrange = None
loss_plot_yrange = [-30, 10]
progress_bar = False
live_visualization = False
 