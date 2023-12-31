"""
original: Stl_TGA_tpl_normtotalflux_40

Settl library
T: 2600~7000 (in log)
g: 2.5 ~ 5 (in log)
Av: 0 ~ 10 mag
for template range: ['l{:d}'.format(i) for i in range(750, 3681-1)]
normalized by total flux (within the corresponding wavelength range)

- modify 'tablename'(path to training database,DB) if you need to use DB

"""
 
# x_names and y_names according to DB (tablename) column names
x_names = [
	'logTeff' ,
	'logG' ,
	'A_V'
]
y_names = ['l{:d}'.format(i) for i in range(750, 3681-1)]
 
# name of output network, Ex) output/xxxx_network.pt
# Do not forget to make upper directory if you use
filename = 'Stl_TGA_tpl.pt'
 
# name (inculuding path) of database
tablename = 'Database/ecogal_spectra_ts_Settl_AV0to10.csv'
 
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
train_smoothing = False
 
# Normalize spectra (option: normalize_total_flux, normalize_mean_flux)
normalize_flux = True
normalize_total_flux = True
normalize_mean_flux = None
 
# Prenoise training (Soft training): Use dY/Y as additional conditions
prenoise_training = False
 
"""
 Training hyperparameters 
"""
device = 'cuda'
# device: cpu, cuda, cuda:0 (modifiable anytime after you read config)
batch_size = 512
n_epochs = 286
 
pre_low_lr = 0
n_its_per_epoch = 2**16
 
load_file = ''
 
checkpoint_save = False
checkpoint_save_interval = 240
checkpoint_save_overwrite = True
 
"""
 Learning Optimization 
"""
gamma = 0.7970856065102272
lr_init = 0.0011877922977586253
l2_weight_reg = 0.0006750745506566001
adam_betas = (0.8, 0.8)
meta_epoch = 6
# how often you change the learning rate (1=every epch)
 
do_rev = False
 
"""
 Model Construction 
"""
n_blocks = 11
 
# subnetwork's internal width, the number of layers
internal_width = 256
internal_layer = 7
init_scale = 0.03
# feature_net (cond_net): (y is converted to feature through cond_net to be used as conditions)
y_dim_features = 256
# dimension used in cond_net 
feature_width = 512
feature_layer = 3
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
 
