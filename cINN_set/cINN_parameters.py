#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 16:35:09 2021

@author: daeun

Default parameters of cINN configuration


Training hyperparameters 
        device
        : this should be modifiable anytime by kernel without modifying the original config file
        
visualization hyperparams
    loss_names = ['L_train',  'L_test', 'lr_train', 'lr_test'] 
    # name used in  terminal printing: in our case lr_ is meaningless (=1)
    loss_plot_yrange = [-50, 10]
    loss_plot_xrange = None
    preview_upscale = 3 not used????
    live_visualization = False
    progress_bar = False # tqdm bar for each batch training

"""

###################### Training setting (only/mostly used in train.py) ######################

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


####################### Learning optimization ####################
# These are used both in train or model but mostly in model

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

# Seed for the random network weight initialisation. The default is 1234.
seed_weight_init = 1234

# Update reverse loss during the training (do_rev = True)
# reverse loss (lr) = mean( (x_true - x_rev)^2), x_rev is from inverse of forward output(z) 
# in the inverse process we can add noise to z: latent noise (sigma of gaussian)
do_rev = False
latent_noise = 0. # 0.05 used in mnist example, 


######################### Model construction (used in model_X.py) ###########################
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


# GLOW and AllInOne
# Coneptually, GLOW has twice as many permutations per affine transformation 
# because the GLOW block does two affine transformations in the block,  
# and the all-in-one block only does one.
# AllInOne uses less memory (esp. for high dimensionality) and run faster than GLOW

#----------------- GLOW coupling block ----------------------
# clamp in exponential component. transffered to 'clamp' in GLOW coupling block
# default in FrEIA is 5.0, but 2.0 is used as default for WFEMP-cINN project
exponent_clamping = 2.0

# permuation or not (AllInOne already contains permuation )
use_permutation = True 

#----------------- AllInOne coupling block ----------------------
# clamp used in AllinOne. default in FrEIA=2.0
affine_clamping = 2.0

# FrEIA default = False
gin_block = False 

# FrEIA default = 1.0
global_affine_init = 0.7 

# FrEIA default = 'SOFTPLUS' 
global_affine_type = 'SOFTPLUS' 

# No permutation turn off in AllInOne. This chooses the method
# True = use random orthogonal matrix
# False = use an actual hard permutation matrix (subset of orthogonal matrices) 
# FrEIA default = False
permute_soft =  True 

# The permutation can be learned by setting > 0
# set roughly to the number of dimensions, or at most to 64, because it becomes increasingly slow
# FrEIA default = 0
learned_householder_permutation = 0 

# Also related to learning permutation
# FrEIA default = False
reverse_permutation = False 


############################# Visualization ##############################
# Visualize loss during training
# name used in  terminal printing: in our case lr_ is meaningless (=1)
# loss_names = ['L_train',  'L_test', 'lr_train', 'lr_test'] 
# only for visualization during train. we save all columns in loss file
loss_names = ['Loss_train_mn',  'Loss_test_mn', 'Loss_train_mdn', 'Loss_test_mdn'] 

# loss curve: every 20 epochs + at last
loss_plot_yrange = [-40, 10]
loss_plot_xrange = None

# preview_upscale = 3 # not used????

# not available right now. not yet modified
live_visualization = False

# tqdm bar for batch training
progress_bar = False 


