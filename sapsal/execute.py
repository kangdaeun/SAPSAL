# 2022. 1. 10. Ver007 FrEIA version 0.1, 0.2

#!/usr/bin/env python
# import sys
import os
import glob
import re
import torch
import torch.nn
import torch.optim
from torch.nn.functional import binary_cross_entropy_with_logits
# from torch.nn.functional import avg_pool2d, interpolate
# from torch.autograd import Variable
import numpy as np
import tqdm
import gc
# import matplotlib
# import matplotlib.pyplot
# matplotlib.use('Agg')

from time import time

from .models import * # ModelAdamGlow
from .data_loader import DataLoader
from .viz import *
from .tools import train_tools
from .tools import test_tools

# Convergence 
CONV_CUT = 1e-5
N_CONV_CHECK = 20

# Divergence
DIVG_CHUNK_SIZE = 7
N_DIVG_CHECK = 30
DIVG_CRI = 0



class dummy_loss(object):
    def item(self):
        return 1.
    
# Functions for domain adaptaion
def BCE(torch_array, ref_value):
    return binary_cross_entropy_with_logits(torch_array, torch.full_like(torch_array, ref_value))
    
def discriminator_loss(D_real, D_fake, label_smoothing=False):
    if label_smoothing:
        ref_fake = 0.1; ref_real = 0.9
    else:
        ref_fake = 0; ref_real = 1
    loss_real = BCE(D_real, ref_real)
    loss_fake = BCE(D_fake, ref_fake)
    
    return loss_real + loss_fake


def adversarial_loss(D_fake, D_real, label_smoothing=False, use_both=True):
    if label_smoothing:
        ref_fake = 0.1; ref_real = 0.9
    else:
        ref_fake = 0; ref_real = 1
    
    # only using fake 
    if use_both:
        return BCE(D_fake, ref_real)+BCE(D_real, ref_fake)
    else:
        return BCE(D_fake, ref_real)   
    # return binary_cross_entropy_with_logits(D_fake, torch.ones_like(D_fake)) + binary_cross_entropy_with_logits(D_real, torch.zeros_like(D_real)) # Discriminator를 속이기
    # return binary_cross_entropy_with_logits(D_fake, torch.ones_like(D_fake)) + binary_cross_entropy_with_logits(D_real, torch.zeros_like(D_real)) # Discriminator를 속이기
    
    
def compute_mmd_rbf(x, y, sigma=1.0):
    """
    x: (N, D) tensor - fake domain features
    y: (M, D) tensor - real domain features
    sigma: float - RBF kernel bandwidth

    Returns: scalar tensor - MMD^2 between x and y
    """
    def rbf_kernel(a, b):
        dist = torch.cdist(a, b, p=2) ** 2
        return torch.exp(-dist / (2 * sigma ** 2))

    K_xx = rbf_kernel(x, x)
    K_yy = rbf_kernel(y, y)
    K_xy = rbf_kernel(x, y)

    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return mmd


def compute_cmd(x, y, n_moments=5):
    """
    x, y: PyTorch tensors of shape (N, D)
    n_moments: int, number of central moments to include (default: 5)

    Returns:
        scalar tensor (0-dim): CMD distance
    """
    assert x.shape[1] == y.shape[1], "Feature dimension mismatch"

    mx = x.mean(dim=0)
    my = y.mean(dim=0)

    sx = x - mx
    sy = y - my

    cmd = torch.norm(mx - my, p=2)

    for i in range(2, n_moments + 1):
        moment_x = torch.mean(sx ** i, dim=0)
        moment_y = torch.mean(sy ** i, dim=0)
        cmd = cmd + torch.norm(moment_x - moment_y, p=2)

    return cmd





# WGAN-GP에서의 Discriminator 손실 계산
def wasserstein_loss(D_real, D_fake):
    return torch.mean(D_fake) - torch.mean(D_real)

# Gradient Penalty 계산
def gradient_penalty(model, real_data, fake_data):
    device = fake_data.device
    # D에 대한 그라디언트를 계산
    epsilon = torch.rand(real_data.size(0), 1).to(device)
    interpolated_data = epsilon * real_data + (1 - epsilon) * fake_data
    interpolated_data.requires_grad_()
    
    D_interpolated = model.da_disc(interpolated_data)
    grad_outputs = torch.ones(D_interpolated.size()).to(device)
    gradients = torch.autograd.grad(outputs=D_interpolated, inputs=interpolated_data,
                                    grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]
    
    # 그라디언트의 L2 norm 계산
    grad_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)
    gradient_penalty = ((grad_norm - 1) ** 2).mean()
    return gradient_penalty

# WGAN-GP 손실 최종 계산
def total_DA_loss( D_real, D_fake, model, real_data, fake_data, lambda_gp=10):
    # Wasserstein loss + Gradient penalty
    penalty = gradient_penalty(model, real_data, fake_data)
    value = wasserstein_loss(D_real, D_fake) + lambda_gp * penalty
    return value

# not yet used
def calculate_adv_factor(loss_adv, loss_cinn):
    # ratio = np.log10(np.abs(loss_cinn.item()/loss_adv.item()))
    ratio = np.abs(loss_adv/loss_cinn)
    if ratio < 1e-6:
        return 0
    else:
        ratio = np.log10(ratio)
    
        if (ratio>-1)*(ratio<1):
            factor = 1
        elif ratio >= 1:
            factor = 10**(np.ceil(-ratio))
        else:
            factor = 10**(-np.floor(ratio))
        return factor
    
# not yet use
# def add_adversarial_loss(c, model, loss_c, features, features_real=None, y_real=None,):
#     # run only for domain_adaptation
    
#     D_fake = model.da_disc(features)
#     if features_real is None:
#         features_real = model.cond_net.features(y_real)
#     D_real = model.da_disc(features_real)
    
#     if c.da_mode =='simple':
#         loss_adv = adversarial_loss(D_fake, D_real, use_both=c.da_adv_both)#.detach()) 
#         factor = calculate_adv_factor(loss_adv*c.lambda_adv, loss_c)
        
#         l_tot = loss_c + factor * loss_adv*c.lambda_adv
#     elif c.da_mode =='WGAN':
#         loss_adv  = total_DA_loss( D_real, D_fake, model, features_real, features, lambda_gp=c.lambda_adv)
#         factor = calculate_adv_factor(loss_adv, loss_c)
#         l_tot = loss_c + factor * loss_adv
        
#     return (l_tot, loss_adv, factor)




"""
Modified train_network (2022.08.30)
- verbose, etc sentences
"""


    
def train_network(c, data=None, verbose=True, max_epoch=1000, resume=False): # c is cINNConfig class variable

    t_start = time()
    
    # print all parameters in the config
    if verbose:
        config_str = ""
        config_str += "==="*30 + "\n"
        config_str += "Config options:\n\n"

        print_list = c.parameter_list
        if len(c.x_names) > 20: print_list.remove('x_names')
        if len(c.y_names) > 20: print_list.remove('y_names')

        for param in print_list:
            if getattr(c, param) is not None:
                config_str += "  {:25}\t{}\n".format(param, getattr(c, param))

        config_str += "==="*30 + "\n"
        print(config_str)

    #%% Prepare model, loss arrays
    model = eval(c.model_code)(c)
    
    # make savedir
    if (not os.path.exists( os.path.dirname(c.filename) )) and ( os.path.dirname(c.filename)!= '' ):
        os.system('mkdir -p '+os.path.dirname(c.filename))
        
    if c.domain_adaptation:
        tracking_dir = os.path.dirname(c.filename)+'/DA_log/'
        if not os.path.exists(tracking_dir):
            os.system('mkdir -p '+tracking_dir)
        
    # make viz
    if verbose:
        viz = TrainVisualizer(c)
        if c.live_visualization:
            viz.visualizer.viz.text('<pre>' + config_str + '</pre>')
            running_box = viz.visualizer.viz.text('<h1 style="color:red">Running</h1>')
             
    if c.load_file:
        model.load(c.load_file, device=c.device)
        
        
    plot_loss = plot_loss_curve_2types # from viz
    if c.domain_adaptation:
        plot_loss = plot_loss_curve_DA
        
    def sample_z(sigma):
    #    return sigma * torch.cuda.FloatTensor(c.batch_size, c.x_dim).normal_()
        return sigma * torch.FloatTensor(c.batch_size, c.x_dim).normal_().to(c.device)
    
    try:
        # epoch_loss_history = []
        
        # 저장해야할 것. 저장은 항상하고 그외에 다루는 걸 config가 조절해야
        # Loss_train_mn, Loss_test_mn, Loss_train_mdn, Loss_test_mdn
        # L_rev_train_mn, L_rev_test_mn, L_rev_train_mdn, L_rev_test_mdn
        loss_header = ['Epoch']
        loss_header +=['Loss_train_mn', 'Loss_test_mn', 'Loss_train_mdn', 'Loss_test_mdn']
        if c.do_rev:
            loss_header += ['L_rev_train_mn', 'L_rev_test_mn', 'L_rev_train_mdn', 'L_rev_test_mdn']
        # if domain adapation
        if c.domain_adaptation:
            loss_header += ['DLoss_train_mn', 'DLoss_test_mn', 'DLoss_train_mdn', 'DLoss_test_mdn', ]

            loss_header += [ 'AdvLoss_train_mn',  'AdvLoss_test_mn', 'AdvLoss_train_mdn', 'AdvLoss_test_mdn']
            
            loss_header += ['NLLLoss_train_mn',  'NLLLoss_test_mn', 'NLLLoss_train_mdn', 'NLLLoss_test_mdn',]
            loss_header += ['Feature_Dist']
                           
    
        loss_format = ['int32'] + ['float32']*len(loss_header)
        epoch_loss_history = np.zeros(max_epoch, 
                                          dtype={'names':loss_header, 'formats':loss_format} )
        epoch_loss_history['Epoch'] = np.arange(max_epoch)
        
        
        # check resume
        if resume==True:
            # check if checkpoint is saved
            if c.checkpoint_save_overwrite: # only one checkpoint saved
                checkpoint_filename = c.filename + '_checkpoint'   
            else:
                checkpoint_files = c.filename + '_checkpoint_*'      #%4i'%i_epoch
                if len(glob.glob( checkpoint_files ))>0:
                    num_file_pairs = [(int(m.group(1)), f) for f in checkpoint_files if (m := re.match(r"checkpoint_(\d+)$", f))]
                    if num_file_pairs:
                        maxnum, checkpoint_filename = max(num_file_pairs)
                    else:
                        resume=False
                else:
                    resume=False
                      
            try:
                model.load(checkpoint_filename, device=c.device)
                print("Loaded from checkpoint (%s)"%checkpoint_filename)
            except:
                resume = False
        else:
            resume=False
            
                
        if data is None:
            data = DataLoader(c, update_rescale_parameters=True)
        
        if model.rescale_params is None or resume==True:
            # pass rescale parameter information
            model.rescale_params = {'mu_x': data.mu_x, 'mu_y':data.mu_y, 'w_x':data.w_x, 'w_y':data.w_y}
            if c.prenoise_training==True:
                model.rescale_params.update( {'mu_s': data.mu_s, 'w_s': data.w_s})
            
        
        ########
        # training status: -1, 0 or 1
        # 0: training in progress (not converged, not diverged)
        # 1: training converged (both train set and test set converged)
        # -1: training diverged (either train set or test set diverged)
        training_status = 0
        i_epoch = - 1
        
        name_check_train = 'Loss_train_mdn'
        name_check_test = 'Loss_test_mdn'
        name_check_train_da = 'DLoss_train_mdn'
        name_check_test_da = 'DLoss_test_mdn'
        
        if c.domain_adaptation:
            additional_domain_adaptation = False
            
        if resume:
            # read loss history and set new i_epoch
            prv_lossfile = c.filename+'_loss_checkpoint.txt'
            prv_loss = np.genfromtxt(prv_lossfile, names=True)
            for col in loss_header[1:]:  # 첫 번째 열 'epoch' 제외
                epoch_loss_history[col][:len(prv_loss)] = prv_loss[col]
            # selected = np.column_stack([prv_loss[col] for col in loss_header])
            # epoch_loss_history[: len(selected)]=selected
            
            i_epoch = epoch_loss_history['Epoch'][len(prv_loss)] -1
            viz.visualizer.counter = i_epoch+1
        
        #%% Training start
        while (training_status == 0 and i_epoch < max_epoch-1):
            epoch_start = time()
            i_epoch += 1
        
            test_loader, train_loader = data.get_loaders( param_seed = i_epoch) # param_seed will randomize parameter if random_parameters is set
        
            data_iter = iter(train_loader)
            test_iter = iter(test_loader)
        
            loss_history = [] # save loss for all batches
            test_loss_history = []

            if c.domain_adaptation: 
                if c.prenoise_training:
                    loader = data.get_da_loaders() # just torch tensor
                    data_real, err_real = loader
                    
                else:
                    data_real = data.get_da_loaders() # just torch tensor
                # loss for domain adaptaion
                d_loss_history = []
                test_d_loss_history = []
                c_loss_history = []
                test_c_loss_history = []
                if c.da_mode=='simple':
                    adv_loss_history = []
                    test_adv_loss_history = []

            if i_epoch < 0:
                for param_group in model.optim.param_groups:
                    param_group['lr'] = c.lr_init * 2e-2
                    
            #%% DA: epoch status check
            if c.domain_adaptation:
                
                # DA_stop = False
                # if c.da_stop_da is not None:
                #     if i_epoch > c.da_warmup + c.da_stop_da:
                #         DA_stop = True # if True, both Disc and Adv is not used : loss and opimmization change
                        
                warming_up = False # if True, it is warming up period: cinn with adv & disc training. very beginning of training
                domain_adaptation = False # if True, domain adaption: cinn+adv (depending on train_setp) & disc training (depend)
                focus_cinn = False # if  True, only cINN w/o adv is being trained. This is after warming up
                
                start_da = c.da_warmup; end_da = max_epoch
                
                # Set situation depending on i_epoch
                if i_epoch < c.da_warmup:
                    warming_up = True
                elif c.da_stop_da is not None:
                    last_da = c.da_warmup + c.da_stop_da - 1
                    if i_epoch < c.da_warmup + c.da_stop_da:
                        domain_adaptation = True
                    else:
                        focus_cinn = True
                else:
                    domain_adaptation = True
                    
              
                # special case   
                if additional_domain_adaptation: 
                    domain_adaptation = True
                    focus_cinn = False; warming_up=False
                    

                # Set loss setting and optimization set up for domain adaptation
                update_main = False; update_disc = False
                add_adv = False; cal_adv = False
                cal_disc = False;
                turnoff_condnet = False # to fix cond_net after DA finished
                turnoff_INN = False # to help modifying cond_net
                
                
                # warmup / DA / cINN
                if warming_up:
                    # WARMUP period
                    cal_adv = False; add_adv = False; update_main=True # update only cINN part
                    if i_epoch >= c.da_disc_warmup_delay: 
                        update_disc=True; cal_disc=True
                    else:
                        update_disc=False; cal_disc=False
                
                else:
                    cal_adv = True; cal_disc = True; # after warmup, always calculate adv loss and disc loss, even if they are not used
                    
                    if domain_adaptation:
                        # Domain Adaptaion after warm up, before DA stop
                        add_adv = True; update_main = True; update_disc = True # basic setting during DA
                        
                        # 일단 DA 동안 아예 classifier 는 업데이트에 반영 안해보기
                        turnoff_INN = True
                        
                        # update of discriminator depends on da_disc_train_step
                        # currently, integer / alternate / 
                        if c.da_disc_train_step == 'alternate': # alternate cinn and disc optimization. #### ignore delay_cinn
                            if i_epoch%2==0: 
                                update_disc = True; update_main = False
                            else: 
                                update_main = True; update_disc = False
                        elif type(c.da_disc_train_step)==int:
                            update_main = True
                            if i_epoch%c.da_disc_train_step == 0:
                                update_disc = True; 
                            else:
                                update_disc = False
                        # But at the very first DA. update both
                        if i_epoch==start_da:
                            update_disc = True; update_main = True
                    elif focus_cinn:
                        # Not DA. But dont update disc. only focus on cINN
                        add_adv = False
                        update_disc = False
                        update_main = True
                        turnoff_condnet = True
                    else: 
                       # Umm. not yet decided
                       add_adv = False
                       update_disc = False
                       update_main = True
                       
                if verbose:
                    status_txt = f"{i_epoch}: Warming-Up: {warming_up}, Domain adaptaion: {domain_adaptation}, Focus cINN: {focus_cinn}"
                    status_txt += f' (cal_adv: {cal_adv}, add_adv: {add_adv}, cal_disc: {cal_disc}, update_main: {update_main}, update_disc: {update_disc})'
                    print(status_txt)
                        
                    
            #%% Train loop
            # Loop for train_loader, Optimization
            model.train()  # In train mode: Necessary if Dropout or BatchNorm is used in the network
            for i_batch, data_tuple in tqdm.tqdm(enumerate(data_iter),
                                                  total=min(len(train_loader), c.n_its_per_epoch),
                                                  leave=False,
                                                  mininterval=1.,
                                                  disable=(not c.progress_bar),
                                                  ncols=83):
    
                x, y = data_tuple
                
                # # Noise-Net: prenoise training => 2025.05.28 This is moved to data_loader. Now alwasy get_loader return rescaled values
                # # In prenoise training, data are in physical scale (param, obs)
                # if c.prenoise_training:
                #     ymin = torch.min(abs(y), dim=1).values # minimum value for each 
                #     # unc_corrl, unc_sampling etc are considerd 
                #     if c.unc_corrl=='Seg_Flux':
                #         flux = y.detach().cpu().numpy()
                #     else: 
                #         flux=None
                #     sig = torch.Tensor( data.create_uncertainty(tuple(y.shape), expand=c.n_sig_MC, flux=flux  ) ) # n_rows = B x n_sig_MC
                #     if c.n_sig_MC*c.n_noise_MC > 1:
                #         # N_mc expansion (n_noise_MC) + noise sampling (all line independent)
                #         x = torch.repeat_interleave(x, c.n_sig_MC*c.n_noise_MC, dim=0) 
                #         y = torch.repeat_interleave(y, c.n_sig_MC*c.n_noise_MC, dim=0)
                #         sig = torch.repeat_interleave(sig, c.n_noise_MC, dim=0)
                #     y = torch.clip( y * (1+ (10**(sig))*torch.randn(y.shape[0], y.shape[1]) ), min=ymin.reshape(-1,1) ) # all line independent
                    
                #     # Transform to rescaled: np array again
                #     x = torch.Tensor(data.params_to_x(x))
                #     y = torch.hstack((torch.Tensor(data.obs_to_y(y)), torch.Tensor(data.unc_to_sig(10**sig))))
                
                x, y = x.to(c.device), y.to(c.device)
      

                # Domain Adaptaion
                # Discriminator loss calculation
                if c.domain_adaptation: 
                    # Curretnly domain adaptaion is not applied to prenoise training
                    # if c.prenoise_training:
                    #     y_real = data_real
                    #     s_real = err_real
                    #     y_real = torch.hstack((torch.Tensor(data.obs_to_y(y_real)), torch.Tensor(data.unc_to_sig(s_real)))).to(c.device)
                    # else:
                    y_real = data_real.to(c.device)
                        
                    if turnoff_condnet:
                        for param in model.cond_net.parameters():
                            param.requires_grad = False 
                            # in this epoch, cond_net is fixed
                    else:
                        for param in model.cond_net.parameters():
                            param.requires_grad = True
                        
                    if turnoff_INN:
                        for param in model.params_trainable_INN:
                            param.requires_grad = False 
                    else:
                        for param in model.params_trainable_INN:
                            param.requires_grad = True
                        
                    if cal_disc:
                        features_fake = model.cond_net.features(y)
                        features_real = model.cond_net.features(y_real)
                        # features_real = features_real.detach()#.clone() 
                        # features_fake = features_fake.detach().clone()
                    
                        D_real = model.da_disc(features_real.detach().clone() ) # detach used to separately train main cinn and discriminator
                        D_fake = model.da_disc(features_fake.detach().clone() ) 
        
                        if c.da_mode=='WGAN':
                            loss_D = total_DA_loss( D_real, D_fake, model, features_real, features_fake, lambda_gp=c.lambda_adv)
                        elif c.da_mode=='simple':
                            loss_D = discriminator_loss(D_real, D_fake, label_smoothing=c.da_label_smoothing)
                        
                    if update_disc:
                        loss_D.backward()
                        model.da_disc.optim_step() # update discriminator (step+zerograd)
                    # features = features.detach().clone()
                    
                # cINN
                features = model.cond_net.features(y)
                output, jac = model.model(x, features)
                
                zz = torch.sum(output**2, dim=1)
                neg_log_likeli = 0.5 * zz - jac
                ll = torch.mean(neg_log_likeli) # cINN loss
                
                
                # Calculate total loss for main network  and make backward              
                if c.domain_adaptation:
                    if cal_adv:  # calculate adversarial loss if needed
                        # Update main loss by Adding adversarial loss
                        # loss_D에 사용되는 모든 값들을 loss_D.backward() 이후로 다시 계산해줄것.
                        D_fake = model.da_disc(features)  # Use updated Discriminator, but do not detach!
                        features_real = model.cond_net.features(y_real)
                        D_real = model.da_disc(features_real) 
                        if c.da_mode =='simple':
                            loss_adv = adversarial_loss(D_fake, D_real, label_smoothing=c.da_label_smoothing, use_both=c.da_adv_both)#.detach()) 
                        elif  c.da_mode =='WGAN':
                            loss_adv  = total_DA_loss( D_real, D_fake, model, features_real, features, lambda_gp=c.lambda_adv)
                            
                    if add_adv:
                        if c.da_mode =='simple':
                            l = ll + c.lambda_adv * loss_adv
                        elif c.da_mode =='WGAN':
                            l = ll + loss_adv
                    else:
                        l = ll
                        
                    if update_main:
                        l.backward(retain_graph=c.do_rev)         
                    
                else:
                    l = ll
                    l.backward(retain_graph=c.do_rev)
    
                if c.do_rev:
                    samples_noisy = sample_z(c.latent_noise) + output.data
                    # original code raised AssertionError (no condition input)
                    # x_rec = model.model(samples_noisy, rev=True)
                    x_rec, _ = model.model(samples_noisy, features, rev=True)
                    l_rev = torch.mean( (x-x_rec)**2 )
                    
                    l_rev.backward()
                else:
                    l_rev = dummy_loss()
    
               
                # Update network parameters
                if c.domain_adaptation:
                    # After all conditions
                    # if update_disc:
                    #     model.da_disc.optim_step() # update discriminator (step+zerograd)
                    if update_main:
                        model.optim_step() 
                else:
                    model.optim_step() 
                    
               
                loss_history.append([l.item(), l_rev.item()])
                if c.domain_adaptation:
                    c_loss_history.append(ll.item())
                    try:
                        d_loss_history.append(loss_D.item())
                    except:
                        d_loss_history.append(np.nan)
                    
                    try:
                        adv_loss_history.append(loss_adv.item())
                    except:
                        adv_loss_history.append(np.nan)
    
                if i_batch+1 >= c.n_its_per_epoch:
                    # somehow the data loader workers don't shut down automatically
                    try:
                        data_iter._shutdown_workers()
                    except:
                        pass
                    break
                
                # if i_batch==0:
                #     # print(f"features_fake avg & std:{features.mean().item():.3f}, {features.var(dim=0).mean().item():.3f}" )
                #     print(f"\t(10,20) value:{features[10, 20].item()}")
                #     _checking.append(features[10, 20].item())
                
            # print(f"({i_epoch}):", _checking)
        
            #%% Test loop
            # Loop for test_loader, Get loss for test set
            model.eval()  #  In evaluation mode: Necessary if Dropout or BatchNorm is used in the network
            for i_batch, test_tuple in tqdm.tqdm(enumerate(test_iter),
                                                  total=min(len(test_loader), c.n_its_per_epoch),
                                                  leave=False,
                                                  mininterval=1.,
                                                  disable=(not c.progress_bar),
                                                  ncols=83):
    
                x, y = test_tuple
                
                x, y = x.to(c.device), y.to(c.device)

                with torch.no_grad():
                    features = model.cond_net.features(y)
                    output, jac = model.model(x, features)
                        
                zz = torch.sum(output**2, dim=1)
                neg_log_likeli = 0.5 * zz - jac
    
                tl_cinn = torch.mean(neg_log_likeli)
                # DO NOT BACKWARD FOR TEST!! USE NO_GRAD()
                # tl.backward(retain_graph=c.do_rev)
                
                # Domain Adaptaion
                # 이미 torch로 있다고 가정. 리스케일까지 되어서, y_real
                if c.domain_adaptation:
                    y_real = data_real.to(c.device) 
                        
    
                    with torch.no_grad():
                        features_real = model.cond_net.features(y_real)
                        D_real = model.da_disc.forward(features_real)
                        D_fake = model.da_disc.forward(features)
                            
                    if cal_disc:
                        if c.da_mode=='simple':
                            loss_D_test = discriminator_loss(D_real, D_fake, label_smoothing=c.da_label_smoothing)
                        elif c.da_mode=='WGAN':
                            loss_D_test = wasserstein_loss(D_real, D_fake) 
                                
                    if cal_adv:
                        if c.da_mode=='simple':
                            loss_adv_test = adversarial_loss(D_fake, D_real, label_smoothing=c.da_label_smoothing, use_both=c.da_adv_both) 
                        elif c.da_mode=='WGAN':
                            loss_adv_test = wasserstein_loss(D_real, D_fake) 
                            
                    if add_adv:
                        if c.da_mode=='simple':
                            tl = tl_cinn + c.lambda_adv * loss_adv_test
                        elif c.da_mode=='WGAN':
                            # loss_D_test = total_DA_loss( D_real, D_fake, model, features_real, features, lambda_gp=10)
                            loss_D_test = wasserstein_loss(D_real, D_fake) 
                            tl = tl_cinn + loss_adv_test
                            
                    
                    else:
                        tl = tl_cinn
                        
                else:
                    tl = tl_cinn
                    
             
                if c.do_rev:
                    samples_noisy = sample_z(c.latent_noise) + output.data
                    with torch.no_grad():
                        x_rec, _ = model.model(samples_noisy, features, rev=True)
                        tl_rev = torch.mean( (x-x_rec)**2 )
                else:
                    tl_rev = dummy_loss()
                # tl_rev = dummy_loss()
    
                test_loss_history.append([tl.item(), tl_rev.item()])
                if c.domain_adaptation:
                    test_c_loss_history.append(tl_cinn.item())
                    try:
                        test_d_loss_history.append(loss_D_test.item())
                    except:
                        test_d_loss_history.append(np.nan)
                    
                    try:
                        test_adv_loss_history.append(loss_adv_test.item())
                    except:
                        test_adv_loss_history.append(np.nan)

                if i_batch+1 >= c.n_its_per_epoch:
                    # somehow the data loader workers don't shut down automatically
                    try:
                        test_iter._shutdown_workers()
                    except:
                        pass
                    break
                
            
            model.train()  # In train mode: Necessary if Dropout or BatchNorm is used in the network
            
            if c.domain_adaptation: 
                if update_main:
                    model.weight_scheduler.step()
            else:        
                model.weight_scheduler.step()   
                
            
            # calculate loss of this epoch
            epoch_losses_mn = np.mean(np.array(loss_history), axis=0) # 이걸 mean 하는지 median하는지에 따라서. 
            test_epoch_losses_mn = np.mean(np.array(test_loss_history), axis=0)
            epoch_losses_mdn = np.median(np.array(loss_history), axis=0)
            test_epoch_losses_mdn = np.median(np.array(test_loss_history), axis=0)
            
            if c.domain_adaptation:
                epoch_D_losses_mn = np.mean(np.array(d_loss_history))
                test_epoch_D_losses_mn = np.mean(np.array(test_d_loss_history))
                epoch_c_losses_mn = np.mean(np.array(c_loss_history))
                test_epoch_c_losses_mn = np.mean(np.array(test_c_loss_history))
                
                epoch_adv_losses_mn = np.mean(np.array(adv_loss_history))
                test_epoch_adv_losses_mn = np.mean(np.array(test_adv_loss_history))
    
                epoch_D_losses_mdn = np.median(np.array(d_loss_history))
                test_epoch_D_losses_mdn = np.median(np.array(test_d_loss_history))
                epoch_c_losses_mdn = np.median(np.array(c_loss_history))
                test_epoch_c_losses_mdn = np.median(np.array(test_c_loss_history))
           
                epoch_adv_losses_mdn = np.median(np.array(adv_loss_history))
                test_epoch_adv_losses_mdn = np.median(np.array(test_adv_loss_history))
            
            epoch_loss_history['Loss_train_mn'][i_epoch] = epoch_losses_mn[0]
            epoch_loss_history['Loss_test_mn'][i_epoch] = test_epoch_losses_mn[0]
            epoch_loss_history['Loss_train_mdn'][i_epoch] = epoch_losses_mdn[0]
            epoch_loss_history['Loss_test_mdn'][i_epoch] = test_epoch_losses_mdn[0]
            if c.do_rev:
                epoch_loss_history['L_rev_train_mn'][i_epoch] = epoch_losses_mn[1]
                epoch_loss_history['L_rev_test_mn'][i_epoch] = test_epoch_losses_mn[1]
                epoch_loss_history['L_rev_train_mdn'][i_epoch] = epoch_losses_mdn[1]
                epoch_loss_history['L_rev_test_mdn'][i_epoch] = test_epoch_losses_mdn[1]
            
            if c.domain_adaptation:
                epoch_loss_history['DLoss_train_mn'][i_epoch] = epoch_D_losses_mn
                epoch_loss_history['DLoss_test_mn'][i_epoch] = test_epoch_D_losses_mn
                epoch_loss_history['DLoss_train_mdn'][i_epoch] = epoch_D_losses_mdn
                epoch_loss_history['DLoss_test_mdn'][i_epoch] = test_epoch_D_losses_mdn
                
                epoch_loss_history['NLLLoss_train_mn'][i_epoch] = epoch_c_losses_mn
                epoch_loss_history['NLLLoss_test_mn'][i_epoch] = test_epoch_c_losses_mn
                epoch_loss_history['NLLLoss_train_mdn'][i_epoch] = epoch_c_losses_mdn
                epoch_loss_history['NLLLoss_test_mdn'][i_epoch] = test_epoch_c_losses_mdn
                                       
                
                epoch_loss_history['AdvLoss_train_mn'][i_epoch] = epoch_adv_losses_mn
                epoch_loss_history['AdvLoss_test_mn'][i_epoch] = test_epoch_adv_losses_mn
                epoch_loss_history['AdvLoss_train_mdn'][i_epoch] = epoch_adv_losses_mdn
                epoch_loss_history['AdvLoss_test_mdn'][i_epoch] = test_epoch_adv_losses_mdn
                
                epoch_loss_history['Feature_Dist'][i_epoch] = test_tools.calculate_domain_distance(model, data)
                ## checking
                # print(epoch_loss_history['Feature_MMD'][i_epoch])
        
            
            if verbose:
                if i_epoch >= 0:
                    viz.show_loss([epoch_loss_history[k_name][i_epoch] for k_name in c.loss_names], 
                                  logscale=False, its=min(len(test_loader), c.n_its_per_epoch) )
            
                    output_orig = output.cpu()
                    viz.show_hist(output_orig) # pass
    
            with torch.no_grad():
                samples = sample_z(1.)
                pass
            
            #%% Loss and status check for i_epoch
            # Loss and status checking
            loss_plot_kwarg={
                'title': os.path.basename(c.filename).replace('.pt',''), 
                'figname': c.filename+'_Loss_plot.pdf', 
                'yrange': c.loss_plot_yrange,
                'xrange': c.loss_plot_xrange, 
                }
            
            if c.domain_adaptation:
                basename = os.path.basename(c.filename)
                
                fd = epoch_loss_history['Feature_Dist'][i_epoch] 
                title = os.path.basename(c.filename).replace('.pt','') + f" (epoch: {i_epoch}, FD: {fd:.3g})"
                
                tf_filename = tracking_dir+f'TestFeautures_{i_epoch:03d}.npy'
                rf_filename = tracking_dir+f'RealFeautures_{i_epoch:03d}.npy'
                Ddist_kwarg = {
                    'figname': tracking_dir+basename+f'_Ddist_{i_epoch:03d}.pdf' ,
                    'title': title,
                    }
                tSNE_kwarg = {
                    'figname': tracking_dir+basename+f'_tSNE_{i_epoch:03d}.pdf' ,
                    'title': title,
                    }
            
            if (i_epoch > 5)*((i_epoch % 10)==0): # general epoch interval for all
                plot_loss(epoch_loss_history[:i_epoch+1], c=c, return_figure=False, **loss_plot_kwarg)
    
                # if c.domain_adaptation:
                #     test_tools.eval_DA_status(model, data,  Ddist_kwarg=Ddist_kwarg, tSNE_kwarg=tSNE_kwarg)
                    # D_test, D_real = test_tools.calculate_D(model, data, smoothing=c.train_smoothing)
                    # test_tools.plot_D_distribution(D_test, D_real, figname=c.filename+f'_Ddist_{i_epoch}.pdf', return_figure=False,
                    #                     title=os.path.basename(c.filename).replace('.pt','') + f" (epoch: {i_epoch})",  )
               
            elif c.domain_adaptation: 
                if warming_up: # save Ddist, tSNE
                    if i_epoch > 2:
                        plot_loss(epoch_loss_history[:i_epoch+1], c=c, return_figure=False, **loss_plot_kwarg)
                    run_tSNE=True; run_Ddist=True
                    test_tools.eval_DA_status(model, data,  Ddist_kwarg=Ddist_kwarg, tSNE_kwarg=tSNE_kwarg, 
                                              tf_filename=tf_filename, rf_filename=rf_filename, run_tSNE=run_tSNE, run_Ddist=run_Ddist)
                
                elif domain_adaptation:
                    # if i_epoch == c.da_warmup: # right after stating DA
                    #     plot_loss(epoch_loss_history[:i_epoch+1], c=c, return_figure=False, **loss_plot_kwarg)
                    #     test_tools.eval_DA_status(model, data,  Ddist_kwarg=Ddist_kwarg, tSNE_kwarg=tSNE_kwarg)
                    # elif i_epoch < c.da_warmup + 10:
                    #     plot_loss(epoch_loss_history[:i_epoch+1], c=c, return_figure=False, **loss_plot_kwarg)
                    #     test_tools.eval_DA_status(model, data,  Ddist_kwarg=Ddist_kwarg, tSNE_kwarg=tSNE_kwarg)
                    # elif i_epoch%2==0:
                    #     plot_loss(epoch_loss_history[:i_epoch+1], c=c, return_figure=False, **loss_plot_kwarg)
                    #     test_tools.eval_DA_status(model, data,  Ddist_kwarg=Ddist_kwarg, tSNE_kwarg=tSNE_kwarg)
                    plot_loss(epoch_loss_history[:i_epoch+1], c=c, return_figure=False, **loss_plot_kwarg)
                    run_tSNE=True; run_Ddist=True
                    test_tools.eval_DA_status(model, data,  Ddist_kwarg=Ddist_kwarg, tSNE_kwarg=tSNE_kwarg, 
                                              tf_filename=tf_filename, rf_filename=rf_filename, run_tSNE=run_tSNE, run_Ddist=run_Ddist)
                    
                elif focus_cinn:
                    if i_epoch==c.da_warmup + c.da_stop_da:
                        plot_loss(epoch_loss_history[:i_epoch+1], c=c, return_figure=False, **loss_plot_kwarg)
                        run_tSNE=True; run_Ddist=True
                        test_tools.eval_DA_status(model, data,  Ddist_kwarg=Ddist_kwarg, tSNE_kwarg=tSNE_kwarg, 
                                                  tf_filename=tf_filename, rf_filename=rf_filename, run_tSNE=run_tSNE, run_Ddist=run_Ddist)
                       
            # Checkpoint
            if c.checkpoint_save:
                if (i_epoch % c.checkpoint_save_interval) == 0 and i_epoch > 0:
                    if c.checkpoint_save_overwrite:
                        checkpoint_filename = c.filename + '_checkpoint'
                    else:
                        checkpoint_filename = c.filename + '_checkpoint_%4i'%i_epoch
                    model.save(checkpoint_filename)
                    # model.save(c.filename + '_checkpoint_%.4i' % (i_epoch * (1-c.checkpoint_save_overwrite)))
                    
                    header = '\t'.join(epoch_loss_history.dtype.names)
                    formats = []
                    for name in epoch_loss_history.dtype.names:
                        if epoch_loss_history[name].dtype == np.int32:
                            formats.append('%d')
                        else:
                            formats.append('%.8f')
                    np.savetxt(c.filename+'_loss_checkpoint.txt', epoch_loss_history[:i_epoch+1], delimiter='\t', fmt='\t'.join(formats), header=header)
                    
            #####
            # Check convergence and divergence and change training_status
            #####
            if i_epoch > N_CONV_CHECK:
                flag_train_conv = train_tools.check_convergence(epoch_loss_history[name_check_train][:i_epoch+1], 
                                                   conv_cut=CONV_CUT, n_conv_check=N_CONV_CHECK)
                flag_test_conv = train_tools.check_convergence(epoch_loss_history[name_check_test][:i_epoch+1],
                                               conv_cut=CONV_CUT, n_conv_check=N_CONV_CHECK)
                if flag_train_conv * flag_test_conv:
                    if verbose:
                        print("Both train and test curves converged at: %d"%i_epoch)
                    training_status = 1 # 1=converged
                    
            if i_epoch > N_DIVG_CHECK:
                negloglike=True
                if c.domain_adaptation: 
                    negloglike=False
                flag_train_divg = train_tools.check_divergence(epoch_loss_history[name_check_train][:i_epoch+1], negloglike=negloglike,
                                                   chunk_size=DIVG_CHUNK_SIZE, n_divg_check=N_DIVG_CHECK, divg_cri=DIVG_CRI)
                flag_test_divg = train_tools.check_divergence(epoch_loss_history[name_check_test][:i_epoch+1], negloglike=negloglike,
                                                  chunk_size=DIVG_CHUNK_SIZE, n_divg_check=N_DIVG_CHECK, divg_cri=DIVG_CRI)
                # if c.domain_adaptation:
                #     if train_tools.check_divergence(epoch_loss_history[name_check_train_da][:i_epoch+1], negloglike=negloglike,
                #                                        chunk_size=DIVG_CHUNK_SIZE, n_divg_check=N_DIVG_CHECK, divg_cri=DIVG_CRI):
                #         flag_train_divg = True
                #     if train_tools.check_divergence(epoch_loss_history[name_check_test_da][:i_epoch+1], negloglike=negloglike,
                #                                       chunk_size=DIVG_CHUNK_SIZE, n_divg_check=N_DIVG_CHECK, divg_cri=DIVG_CRI):
                #         flag_test_divg = True
                    
                if flag_test_divg==True or flag_train_divg==True:
                    if verbose:
                        print("Either train or test curve diverged at: %d"%i_epoch)
                    training_status = -1 # -1 = diverged
                    
            print(f"[Epoch {i_epoch}] Time: {time() - epoch_start:.2f}s")
            gc.collect()
            torch.cuda.empty_cache()
        
        #%% End of all epochs 
        ### END LOOP    
        final_epoch = i_epoch + 1
        # remove unnecessary data in loss 
        epoch_loss_history = epoch_loss_history[:final_epoch]
        # update config with actural n_epochs used
        c.n_epochs = final_epoch
        train_tools.rewrite_config_element(c.config_file,  new_config_file=c.config_file, 
                                           param_to_change='n_epochs', value_to_change=c.n_epochs)
        
        if verbose:
            if c.live_visualization:
                viz.visualizer.viz.text('<h1 style="color:green">Done</h1>', win=running_box)
        model.save(c.filename)
        
        # move domain adapation tracking plots
        # if c.domain_adaptation:
        #     tracking_dir = os.path.dirname(c.filename)+'/DA_log/'
        #     if not os.path.exists(tracking_dir):
        #         os.system('mkdir -p '+tracking_dir)
        #     os.system('mv {}/*Ddist_*.pdf {}'.format(os.path.dirname(c.filename), tracking_dir))
        #     os.system('mv {}/*tSNE_*.pdf {}'.format(os.path.dirname(c.filename), tracking_dir))
            
        
        
        
        # Plot final loss
        loss_plot_kwarg={
            'title': os.path.basename(c.filename).replace('.pt',''), 
            'figname': c.filename+'_Loss_plot.pdf', 
            'yrange': c.loss_plot_yrange,
            'xrange': c.loss_plot_xrange, 
            }
        plot_loss(epoch_loss_history, c=c, return_figure=False, **loss_plot_kwarg)
        
        if c.domain_adaptation:
            basename = os.path.basename(c.filename)
            fd = epoch_loss_history['Feature_Dist'][final_epoch-1] 
            title = os.path.basename(c.filename).replace('.pt','') + f" (epoch: {i_epoch}, FD: {fd:.3g})"
            tf_filename = tracking_dir+f'TestFeautures_{i_epoch:03d}.npy'
            rf_filename = tracking_dir+f'RealFeautures_{i_epoch:03d}.npy'
            Ddist_kwarg = {
                'figname':c.filename+'_Ddist.pdf' ,
                'title': title,
                }
            tSNE_kwarg = {
                'figname': c.filename+'_tSNE.pdf' ,
                'title': title,
                }
            test_tools.eval_DA_status(model, data,  Ddist_kwarg=Ddist_kwarg, tSNE_kwarg=tSNE_kwarg, 
                                     tf_filename=tf_filename, rf_filename=rf_filename, run_tSNE=True, run_Ddist=True)
        

        header = '\t'.join(epoch_loss_history.dtype.names)
        formats = []
        for name in epoch_loss_history.dtype.names:
            if epoch_loss_history[name].dtype == np.int32:
                formats.append('%d')
            else:
                formats.append('%.8f')
        np.savetxt(c.filename+'_loss_history.txt', epoch_loss_history, delimiter='\t', fmt='\t'.join(formats), header=header)
        
        if c.checkpoint_save and c.checkpoint_remove_after_training:
            checkpoint_files = c.filename + '_checkpoint*'      #%4i'%i_epoch
            if len(glob.glob( checkpoint_files ))>0:
                os.system('rm '+checkpoint_files)
            
            if os.path.exists( c.filename+'_loss_checkpoint.txt'):
                os.system('rm '+c.filename+'_loss_checkpoint.txt')

    except:
        #%% Aborted
        model.save(c.filename + '_ABORT')
        
        header = '\t'.join(epoch_loss_history.dtype.names)
        formats = []
        for name in epoch_loss_history.dtype.names:
            if epoch_loss_history[name].dtype == np.int32:
                formats.append('%d')
            else:
                formats.append('%.8f')
        np.savetxt(c.filename+'_loss_history_ABORT.txt', epoch_loss_history, delimiter='\t', fmt='\t'.join(formats), header=header)
        
        if len(epoch_loss_history) > 10:
            loss_plot_kwarg={
                'title': os.path.basename(c.filename).replace('.pt',''), 
                'figname': c.filename+'_ABORT_Loss_plot.pdf', 
                'yrange': c.loss_plot_yrange,
                'xrange': c.loss_plot_xrange, 
                }
            plot_loss(epoch_loss_history, c=c, return_figure=False, **loss_plot_kwarg)
           

        raise
    
    t_end = time()
    print('Time taken for training: {:.1f} hour ({:.1f} min)'.format( (t_end-t_start)/3600, (t_end-t_start)/60 ) )
    # return epoch_loss_history
    
    



def train_network_DAwoD(c, data=None, verbose=True, max_epoch=1000): # c is cINNConfig class variable

    t_start = time()
    
    # print all parameters in the config
    if verbose:
        config_str = ""
        config_str += "==="*30 + "\n"
        config_str += "Config options:\n\n"

        print_list = c.parameter_list
        if len(c.x_names) > 20: print_list.remove('x_names')
        if len(c.y_names) > 20: print_list.remove('y_names')

        for param in print_list:
            if getattr(c, param) is not None:
                config_str += "  {:25}\t{}\n".format(param, getattr(c, param))

        config_str += "==="*30 + "\n"
        print(config_str)

    #%% Prepare model, loss arrays
    model = eval(c.model_code)(c)
    
    # make savedir
    if (not os.path.exists( os.path.dirname(c.filename) )) and ( os.path.dirname(c.filename)!= '' ):
        os.system('mkdir -p '+os.path.dirname(c.filename))
        
    tracking_dir = os.path.dirname(c.filename)+'/DA_log/'
    if not os.path.exists(tracking_dir):
        os.system('mkdir -p '+tracking_dir)
        
    # make viz
    if verbose:
        viz = TrainVisualizer(c)
        if c.live_visualization:
            viz.visualizer.viz.text('<pre>' + config_str + '</pre>')
            running_box = viz.visualizer.viz.text('<h1 style="color:red">Running</h1>')
             
    if c.load_file:
        model.load(c.load_file, device=c.device)
        
    plot_loss = plot_loss_curve_DAwoD
        
    def sample_z(sigma):
    #    return sigma * torch.cuda.FloatTensor(c.batch_size, c.x_dim).normal_()
        return sigma * torch.FloatTensor(c.batch_size, c.x_dim).normal_().to(c.device)
    
    
    try:
        # epoch_loss_history = []
        
        # 저장해야할 것. 저장은 항상하고 그외에 다루는 걸 config가 조절해야
        # Loss_train_mn, Loss_test_mn, Loss_train_mdn, Loss_test_mdn
        # L_rev_train_mn, L_rev_test_mn, L_rev_train_mdn, L_rev_test_mdn
        loss_header = ['Epoch']
        loss_header +=['Loss_train_mn', 'Loss_test_mn', 'Loss_train_mdn', 'Loss_test_mdn']
        if c.do_rev:
            loss_header += ['L_rev_train_mn', 'L_rev_test_mn', 'L_rev_train_mdn', 'L_rev_test_mdn']
       
       
        loss_header += ['NLLLoss_train_mn',  'NLLLoss_test_mn', 'NLLLoss_train_mdn', 'NLLLoss_test_mdn',]
        loss_header += ['Feature_Dist']
                           
    
        loss_format = ['int32'] + ['float32']*len(loss_header)
        epoch_loss_history = np.zeros(max_epoch, 
                                          dtype={'names':loss_header, 'formats':loss_format} )
        epoch_loss_history['Epoch'] = np.arange(max_epoch)
        
        if data is None:
            data = DataLoader(c, update_rescale_parameters=True)
        
        if model.rescale_params is None:
            # pass rescale parameter information
            model.rescale_params = {'mu_x': data.mu_x, 'mu_y':data.mu_y, 'w_x':data.w_x, 'w_y':data.w_y}
            
        
        ########
        # training status: -1, 0 or 1
        # 0: training in progress (not converged, not diverged)
        # 1: training converged (both train set and test set converged)
        # -1: training diverged (either train set or test set diverged)
        training_status = 0
        i_epoch = - 1
        
        name_check_train = 'Loss_train_mdn'
        name_check_test = 'Loss_test_mdn'
        
        
        stop_domain_checking = False
        #%% Training start
        while (training_status == 0 and i_epoch < max_epoch-1):
            i_epoch += 1
        
            test_loader, train_loader = data.get_loaders( param_seed = i_epoch) # param_seed will randomize parameter if random_parameters is set
        
            data_iter = iter(train_loader)
            test_iter = iter(test_loader)
        
            loss_history = [] # save loss for all batches
            test_loss_history = []

            # DA
            if c.prenoise_training:
                loader = data.get_da_loaders() # just torch tensor
                data_real, err_real = loader
                
            else:
                data_real = data.get_da_loaders() # just torch tensor
            
            # loss for domain adaptaion
            c_loss_history = []
            test_c_loss_history = []

            if i_epoch < 0:
                for param_group in model.optim.param_groups:
                    param_group['lr'] = c.lr_init * 2e-2
                    
            #%% DA: epoch status check
            
                # if verbose:
                #     status_txt = f"{i_epoch}: Warming-Up: {warming_up}, Domain adaptaion: {domain_adaptation}, Focus cINN: {focus_cinn}"
                #     status_txt += f' (cal_adv: {cal_adv}, add_adv: {add_adv}, cal_disc: {cal_disc}, update_main: {update_main}, update_disc: {update_disc})'
                #     print(status_txt)
                
                        
                    
            #%% Train loop
            # Loop for train_loader, Optimization
            model.train()  # In train mode: Necessary if Dropout or BatchNorm is used in the network
            for i_batch, data_tuple in tqdm.tqdm(enumerate(data_iter),
                                                  total=min(len(train_loader), c.n_its_per_epoch),
                                                  leave=False,
                                                  mininterval=1.,
                                                  disable=(not c.progress_bar),
                                                  ncols=83):
    
                x, y = data_tuple
                
                # Noise-Net: prenoise training
                # In prenoise training, data are in physical scale (param, obs)
                if c.prenoise_training:
                    ymin = torch.min(abs(y), dim=1).values # minimum value for each 
                    # unc_corrl, unc_sampling etc are considerd 
                    if c.unc_corrl=='Seg_Flux':
                        flux = y.detach().cpu().numpy()
                    else: 
                        flux=None
                    sig = torch.Tensor( data.create_uncertainty(tuple(y.shape), expand=c.n_sig_MC, flux=flux  ) ) # n_rows = B x n_sig_MC
                    if c.n_sig_MC*c.n_noise_MC > 1:
                        # N_mc expansion (n_noise_MC) + noise sampling (all line independent)
                        x = torch.repeat_interleave(x, c.n_sig_MC*c.n_noise_MC, dim=0) 
                        y = torch.repeat_interleave(y, c.n_sig_MC*c.n_noise_MC, dim=0)
                        sig = torch.repeat_interleave(sig, c.n_noise_MC, dim=0)
                    y = torch.clip( y * (1+ (10**(sig))*torch.randn(y.shape[0], y.shape[1]) ), min=ymin.reshape(-1,1) ) # all line independent
                    
                    # Transform to rescaled: np array again
                    x = torch.Tensor(data.params_to_x(x))
                    y = torch.hstack((torch.Tensor(data.obs_to_y(y)), torch.Tensor(data.unc_to_sig(10**sig))))
                
                x, y = x.to(c.device), y.to(c.device)
                # features = model.cond_net.features(y)

                # Domain Adaptaion
                if c.prenoise_training:
                    y_real = data_real
                    s_real = err_real
                    if c.n_sig_MC*c.n_noise_MC > 1:
                        y_real = torch.repeat_interleave(y_real, c.n_sig_MC*c.n_noise_MC, dim=0) 
                        s_real = torch.repeat_interleave(s_real, c.n_sig_MC*c.n_noise_MC, dim=0) 
                    y_real = torch.hstack((torch.Tensor(data.obs_to_y(y_real)), torch.Tensor(data.unc_to_sig(s_real)))).to(c.device)
                else:
                    y_real = data_real.to(c.device)
                    
                if stop_domain_checking:
                    for param in model.cond_net.parameters():
                        param.requires_grad = False 
                        # from this epoch, cond_net is fixed
               
                    
                # cINN
                features = model.cond_net.features(y)
                output, jac = model.model(x, features)
                
                zz = torch.sum(output**2, dim=1)
                neg_log_likeli = 0.5 * zz - jac
                ll = torch.mean(neg_log_likeli) # cINN loss
                
                
                # DA: calculate feature distance
                features_real = model.cond_net.features(y_real)
                if not stop_domain_checking:
                    # prv_dist = epoch_loss_history['Feature_Dist'][i_epoch-1]
                    domain_distance = compute_cmd(features, features_real)
                    if i_epoch==0:
                        factor = 0
                    else:
                        factor = c.lambda_adv #calculate_adv_factor(prv_dist, ll.item()) 
                    # l = ll + prv_dist * factor
                    l = ll + domain_distance * factor
                else:
                    l = ll
                   
                l.backward(retain_graph=c.do_rev)
    
                if c.do_rev:
                    samples_noisy = sample_z(c.latent_noise) + output.data
                    # original code raised AssertionError (no condition input)
                    # x_rec = model.model(samples_noisy, rev=True)
                    x_rec, _ = model.model(samples_noisy, features, rev=True)
                    l_rev = torch.mean( (x-x_rec)**2 )
                    
                    l_rev.backward()
                else:
                    l_rev = dummy_loss()
    

                model.optim_step() 
                    
               
                loss_history.append([l.item(), l_rev.item()])

                c_loss_history.append(ll.item())
                   
    
                if i_batch+1 >= c.n_its_per_epoch:
                    # somehow the data loader workers don't shut down automatically
                    try:
                        data_iter._shutdown_workers()
                    except:
                        pass
                    break
                
                
        
            #%% Test loop
            # Loop for test_loader, Get loss for test set
            model.eval()  #  In evaluation mode: Necessary if Dropout or BatchNorm is used in the network
            for i_batch, test_tuple in tqdm.tqdm(enumerate(test_iter),
                                                  total=min(len(test_loader), c.n_its_per_epoch),
                                                  leave=False,
                                                  mininterval=1.,
                                                  disable=(not c.progress_bar),
                                                  ncols=83):
    
                x, y = test_tuple

                if c.prenoise_training:
                    ymin = torch.min(abs(y), dim=1).values # minimum value for each 
                    # unc_corrl, unc_sampling etc are considerd 
                    sig = torch.Tensor( data.create_uncertainty(tuple(y.shape), expand=c.n_sig_MC ) ) # n_rows = B x n_sig_MC
                    if c.n_sig_MC*c.n_noise_MC > 1:
                        # N_mc expansion (n_noise_MC) + noise sampling (all line independent)
                        x = torch.repeat_interleave(x, c.n_sig_MC*c.n_noise_MC, dim=0) 
                        y = torch.repeat_interleave(y, c.n_sig_MC*c.n_noise_MC, dim=0)
                        sig = torch.repeat_interleave(sig, c.n_noise_MC, dim=0)
                    y = torch.clip( y * (1+ (10**(sig))*torch.randn(y.shape[0], y.shape[1]) ), min=ymin.reshape(-1,1) ) # all line independent
                    
                    # Transform to rescaled: np array again
                    x = torch.Tensor(data.params_to_x(x))
                    y = torch.hstack((torch.Tensor(data.obs_to_y(y)), torch.Tensor(data.unc_to_sig(10**sig))))
                
                x, y = x.to(c.device), y.to(c.device)

                with torch.no_grad():
                    features = model.cond_net.features(y)
                    output, jac = model.model(x, features)
                        
                zz = torch.sum(output**2, dim=1)
                neg_log_likeli = 0.5 * zz - jac
    
                tl_cinn = torch.mean(neg_log_likeli)
                # DO NOT BACKWARD FOR TEST!! USE NO_GRAD()
                # tl.backward(retain_graph=c.do_rev)
                
                # Domain Adaptaion
                # 이미 torch로 있다고 가정. 리스케일까지 되어서, y_real
              
                if c.prenoise_training:
                    y_real = data_real
                    s_real = err_real
                    if c.n_sig_MC*c.n_noise_MC > 1:
                        y_real = torch.repeat_interleave(y_real, c.n_sig_MC*c.n_noise_MC, dim=0) 
                        s_real = torch.repeat_interleave(s_real, c.n_sig_MC*c.n_noise_MC, dim=0) 
                    y_real = torch.hstack((torch.Tensor(data.obs_to_y(y_real)), torch.Tensor(data.unc_to_sig(s_real)))).to(c.device)
                else:
                    y_real = data_real.to(c.device) 
                    
    
                with torch.no_grad():
                    features_real = model.cond_net.features(y_real)
                
                if not stop_domain_checking:
                    # domain_distance = compute_mmd_rbf(features, features_real)
                    domain_distance = compute_cmd(features, features_real)
                    # prv_dist = epoch_loss_history['Feature_Dist'][i_epoch-1]
                    if i_epoch==0:
                        factor = 0
                    else:
                        factor = c.lambda_adv #calculate_adv_factor(domain_distance.item(), tl_cinn.item()) 
                    # tl = tl_cinn + prv_dist * factor
                    tl = tl_cinn + domain_distance * factor
                else:
                    tl = tl_cinn
                # domain_distance = compute_cmd(features, features_real)
                # tl = tl_cinn + domain_distance
             
                if c.do_rev:
                    samples_noisy = sample_z(c.latent_noise) + output.data
                    with torch.no_grad():
                        x_rec, _ = model.model(samples_noisy, features, rev=True)
                        tl_rev = torch.mean( (x-x_rec)**2 )
                else:
                    tl_rev = dummy_loss()
                # tl_rev = dummy_loss()
    
                test_loss_history.append([tl.item(), tl_rev.item()])
               
                test_c_loss_history.append(tl_cinn.item())
                    

                if i_batch+1 >= c.n_its_per_epoch:
                    # somehow the data loader workers don't shut down automatically
                    try:
                        test_iter._shutdown_workers()
                    except:
                        pass
                    break
                
            
            model.train()  # In train mode: Necessary if Dropout or BatchNorm is used in the network
            model.weight_scheduler.step()   
                
            
            # calculate loss of this epoch
            epoch_losses_mn = np.mean(np.array(loss_history), axis=0) # 이걸 mean 하는지 median하는지에 따라서. 
            test_epoch_losses_mn = np.mean(np.array(test_loss_history), axis=0)
            epoch_losses_mdn = np.median(np.array(loss_history), axis=0)
            test_epoch_losses_mdn = np.median(np.array(test_loss_history), axis=0)
            
            
               
            epoch_c_losses_mn = np.mean(np.array(c_loss_history))
            test_epoch_c_losses_mn = np.mean(np.array(test_c_loss_history))
            
            epoch_c_losses_mdn = np.median(np.array(c_loss_history))
            test_epoch_c_losses_mdn = np.median(np.array(test_c_loss_history))
           
    
            epoch_loss_history['Loss_train_mn'][i_epoch] = epoch_losses_mn[0]
            epoch_loss_history['Loss_test_mn'][i_epoch] = test_epoch_losses_mn[0]
            epoch_loss_history['Loss_train_mdn'][i_epoch] = epoch_losses_mdn[0]
            epoch_loss_history['Loss_test_mdn'][i_epoch] = test_epoch_losses_mdn[0]
            if c.do_rev:
                epoch_loss_history['L_rev_train_mn'][i_epoch] = epoch_losses_mn[1]
                epoch_loss_history['L_rev_test_mn'][i_epoch] = test_epoch_losses_mn[1]
                epoch_loss_history['L_rev_train_mdn'][i_epoch] = epoch_losses_mdn[1]
                epoch_loss_history['L_rev_test_mdn'][i_epoch] = test_epoch_losses_mdn[1]
            
            
               
            epoch_loss_history['NLLLoss_train_mn'][i_epoch] = epoch_c_losses_mn
            epoch_loss_history['NLLLoss_test_mn'][i_epoch] = test_epoch_c_losses_mn
            epoch_loss_history['NLLLoss_train_mdn'][i_epoch] = epoch_c_losses_mdn
            epoch_loss_history['NLLLoss_test_mdn'][i_epoch] = test_epoch_c_losses_mdn
                                       
              
            epoch_loss_history['Feature_Dist'][i_epoch] = test_tools.calculate_domain_distance(model, data, option='CMD')
                ## checking
                # print(epoch_loss_history['Feature_MMD'][i_epoch])
        
            
            if verbose:
                if i_epoch >= 0:
                    viz.show_loss([epoch_loss_history[k_name][i_epoch] for k_name in c.loss_names], 
                                  logscale=False, its=min(len(test_loader), c.n_its_per_epoch) )
            
                    output_orig = output.cpu()
                    viz.show_hist(output_orig) # pass
    
            with torch.no_grad():
                samples = sample_z(1.)
                pass
            
            #%% Loss and status check for i_epoch
            # Loss and status checking
            loss_plot_kwarg={
                'title': os.path.basename(c.filename).replace('.pt',''), 
                'figname': c.filename+'_Loss_plot.pdf', 
                'yrange': c.loss_plot_yrange,
                'xrange': c.loss_plot_xrange, 
                }
            
           
            basename = os.path.basename(c.filename)
                
            fd = epoch_loss_history['Feature_Dist'][i_epoch] 
            title = os.path.basename(c.filename).replace('.pt','') + f" (epoch: {i_epoch}, FD: {fd:.3g})"
                
            tf_filename = tracking_dir+f'TestFeautures_{i_epoch:03d}.npy'
            rf_filename = tracking_dir+f'RealFeautures_{i_epoch:03d}.npy'
              
            tSNE_kwarg = {
                'figname': tracking_dir+basename+f'_tSNE_{i_epoch:03d}.pdf' ,
                'title': title,
                }
            
            if (i_epoch > 5)*((i_epoch % 10)==0): # general epoch interval for all
                plot_loss(epoch_loss_history[:i_epoch+1], c=c, return_figure=False, **loss_plot_kwarg)
    
               
            if stop_domain_checking==False and i_epoch%5==0:
                plot_loss(epoch_loss_history[:i_epoch+1], c=c, return_figure=False, **loss_plot_kwarg)
                test_tools.eval_DA_status(model, data,  tSNE_kwarg=tSNE_kwarg, 
                                          tf_filename=tf_filename, rf_filename=rf_filename, run_tSNE=True, run_Ddist=False)
            
            if c.checkpoint_save:
                if (i_epoch % c.checkpoint_save_interval) == 0:
                    model.save(c.filename + '_checkpoint_%.4i' % (i_epoch * (1-c.checkpoint_save_overwrite)))
                    
                    
            # Check feature distance convergence
            # if not stop_domain_checking:
            #     flag_fd_conv = train_tools.check_convergence(epoch_loss_history['Feature_Dist'][:i_epoch+1], 
            #                                                  conv_cut=1e-3, n_conv_check=10)
            #     if flag_fd_conv:
            #         stop_domain_checking = True
            #     if np.nanmean(epoch_loss_history['Feature_Dist'][i_epoch-3:i_epoch+1]) < 1e-6: 
            #         stop_domain_checking = True
            #     if stop_domain_checking:   
            #         print(f"epoch:{i_epoch:03d} Stop domain checking. Cond-Net is fixed")
                
                    
            #####
            # Check convergence and divergence and change training_status
            #####
            if i_epoch > N_CONV_CHECK:
                flag_train_conv = train_tools.check_convergence(epoch_loss_history[name_check_train][:i_epoch+1], 
                                                   conv_cut=CONV_CUT, n_conv_check=N_CONV_CHECK)
                flag_test_conv = train_tools.check_convergence(epoch_loss_history[name_check_test][:i_epoch+1],
                                               conv_cut=CONV_CUT, n_conv_check=N_CONV_CHECK)
                
                
                if flag_train_conv * flag_test_conv:
                    if verbose:
                        print("Both train and test curves converged at: %d"%i_epoch)
                    training_status = 1 # 1=converged
                    
            if i_epoch > N_DIVG_CHECK:
                negloglike=True
                if c.domain_adaptation: 
                    negloglike=False
                flag_train_divg = train_tools.check_divergence(epoch_loss_history[name_check_train][:i_epoch+1], negloglike=negloglike,
                                                   chunk_size=DIVG_CHUNK_SIZE, n_divg_check=N_DIVG_CHECK, divg_cri=DIVG_CRI)
                flag_test_divg = train_tools.check_divergence(epoch_loss_history[name_check_test][:i_epoch+1], negloglike=negloglike,
                                                  chunk_size=DIVG_CHUNK_SIZE, n_divg_check=N_DIVG_CHECK, divg_cri=DIVG_CRI)
               
                    
                if flag_test_divg==True or flag_train_divg==True:
                    if verbose:
                        print("Either train or test curve diverged at: %d"%i_epoch)
                    training_status = -1 # -1 = diverged
        
        #%% End of all epochs 
        ### END LOOP    
        final_epoch = i_epoch + 1
        # remove unnecessary data in loss 
        epoch_loss_history = epoch_loss_history[:final_epoch]
        # update config with actural n_epochs used
        c.n_epochs = final_epoch
        train_tools.rewrite_config_element(c.config_file,  new_config_file=c.config_file, 
                                           param_to_change='n_epochs', value_to_change=c.n_epochs)
        
        if verbose:
            if c.live_visualization:
                viz.visualizer.viz.text('<h1 style="color:green">Done</h1>', win=running_box)
        model.save(c.filename)
        
       
        
        
        # Plot final loss
        loss_plot_kwarg={
            'title': os.path.basename(c.filename).replace('.pt',''), 
            'figname': c.filename+'_Loss_plot.pdf', 
            'yrange': c.loss_plot_yrange,
            'xrange': c.loss_plot_xrange, 
            }
        plot_loss(epoch_loss_history, c=c, return_figure=False, **loss_plot_kwarg)
        
        basename = os.path.basename(c.filename)
        fd = epoch_loss_history['Feature_Dist'][final_epoch-1] 
        title = os.path.basename(c.filename).replace('.pt','') + f" (epoch: {i_epoch}, FD: {fd:.3g})"
        tf_filename = tracking_dir+f'TestFeautures_{i_epoch:03d}.npy'
        rf_filename = tracking_dir+f'RealFeautures_{i_epoch:03d}.npy'
       
        tSNE_kwarg = {
            'figname': c.filename+'_tSNE.pdf' ,
            'title': title,
            }
        test_tools.eval_DA_status(model, data, tSNE_kwarg=tSNE_kwarg, 
                                 tf_filename=tf_filename, rf_filename=rf_filename, run_tSNE=True, run_Ddist=False)
        

        header = '\t'.join(epoch_loss_history.dtype.names)
        formats = []
        for name in epoch_loss_history.dtype.names:
            if epoch_loss_history[name].dtype == np.int32:
                formats.append('%d')
            else:
                formats.append('%.8f')
        np.savetxt(c.filename+'_loss_history.txt', epoch_loss_history, delimiter='\t', fmt='\t'.join(formats), header=header)

    except:
        #%% Aborted
        model.save(c.filename + '_ABORT')
        
        header = '\t'.join(epoch_loss_history.dtype.names)
        formats = []
        for name in epoch_loss_history.dtype.names:
            if epoch_loss_history[name].dtype == np.int32:
                formats.append('%d')
            else:
                formats.append('%.8f')
        np.savetxt(c.filename+'_loss_history_ABORT.txt', epoch_loss_history, delimiter='\t', fmt='\t'.join(formats), header=header)
        
        if len(epoch_loss_history) > 10:
            loss_plot_kwarg={
                'title': os.path.basename(c.filename).replace('.pt',''), 
                'figname': c.filename+'_ABORT_Loss_plot.pdf', 
                'yrange': c.loss_plot_yrange,
                'xrange': c.loss_plot_xrange, 
                }
            plot_loss(epoch_loss_history, c=c, return_figure=False, **loss_plot_kwarg)
           

        raise
    
    t_end = time()
    print('Time taken for training: {:.1f} hour ({:.1f} min)'.format( (t_end-t_start)/3600, (t_end-t_start)/60 ) )
    # return epoch_loss_history

    

def train_prenoise_network(c, data=None, verbose=True, max_epoch=1000): # c is cINNConfig class variable

    t_start = time()
    
    # print all parameters in the config
    if verbose:
        config_str = ""
        config_str += "==="*30 + "\n"
        config_str += "Config options:\n\n"

        print_list = c.parameter_list
        if len(c.x_names) > 20: print_list.remove('x_names')
        if len(c.y_names) > 20: print_list.remove('y_names')

        for param in print_list:
            if getattr(c, param) is not None:
                config_str += "  {:25}\t{}\n".format(param, getattr(c, param))

        config_str += "==="*30 + "\n"
        print(config_str)

    model = eval(c.model_code)(c)
    
    # make savedir
    if (not os.path.exists( os.path.dirname(c.filename) )) and ( os.path.dirname(c.filename)!= '' ):
        os.system('mkdir -p '+os.path.dirname(c.filename))
        
    # make viz
    if verbose:
        viz = TrainVisualizer(c)
        if c.live_visualization:
            viz.visualizer.viz.text('<pre>' + config_str + '</pre>')
            running_box = viz.visualizer.viz.text('<h1 style="color:red">Running</h1>')
           
        
    if c.load_file:
        model.load(c.load_file, device=c.device)
        
        
    def sample_z(sigma):
    #    return sigma * torch.cuda.FloatTensor(c.batch_size, c.x_dim).normal_()
        return sigma * torch.FloatTensor(c.batch_size, c.x_dim).normal_().to(c.device)
    
    try:
        epoch_loss_history = []
        
        # 저장해야할 것. 저장은 항상하고 그외에 다루는 걸 config가 조절해야
        # Loss_train_mn, Loss_test_mn, Loss_train_mdn, Loss_test_mdn
        # L_rev_train_mn, L_rev_test_mn, L_rev_train_mdn, L_rev_test_mdn
        loss_header = ['Epoch']
        loss_header +=['Loss_train_mn', 'Loss_test_mn', 'Loss_train_mdn', 'Loss_test_mdn']
        # if c.do_rev:
        loss_header += ['L_rev_train_mn', 'L_rev_test_mn', 'L_rev_train_mdn', 'L_rev_test_mdn']

        loss_format = ['int32'] + ['float32']*len(loss_header)
        epoch_loss_history = np.zeros(max_epoch, 
                                          dtype={'names':loss_header, 'formats':loss_format} )
        epoch_loss_history['Epoch'] = np.arange(max_epoch)
        
        if data is None:
            data = DataLoader(c, update_rescale_parameters=True)
        
        if model.rescale_params is None:
            # pass rescale parameter information
#             model.rescale_params = {'mu_x': data.mu_x, 'mu_y':data.mu_y, 'w_x':data.w_x, 'w_y':data.w_y}
            model.rescale_params = {'mu_x': data.mu_x, 'mu_y':data.mu_y, 'w_x':data.w_x, 'w_y':data.w_y, 'mu_s': data.mu_s, 'w_s': data.w_s}
            
        
        ########
        # training status: -1, 0 or 1
        # 0: training in progress (not converged, not diverged)
        # 1: training converged (both train set and test set converged)
        # -1: training diverged (either train set or test set diverged)
        training_status = 0
        i_epoch = - 1
        
        name_check_train = 'Loss_train_mdn'
        name_check_test = 'Loss_test_mdn'
        
        while (training_status == 0 and i_epoch < max_epoch-1):
            i_epoch += 1
            
            test_loader, train_loader = data.get_loaders(param_seed = i_epoch) # param_seed will randomize parameter if random_parameters is set
        
            data_iter = iter(train_loader)
            test_iter = iter(test_loader)
    
            loss_history = [] # save loss for all batches
            test_loss_history = []
            
            if i_epoch < 0:
                for param_group in model.optim.param_groups:
                    param_group['lr'] = c.lr_init * 2e-2
                    
            # Loop for train_loader, Optimization
            for i_batch, data_tuple in tqdm.tqdm(enumerate(data_iter),
                                                  total=min(len(train_loader), c.n_its_per_epoch),
                                                  leave=False,
                                                  mininterval=1.,
                                                  disable=(not c.progress_bar),
                                                  ncols=83):
                
                # In prenoise training, data are in physical scale (param, obs)
                x, y = data_tuple
                
                ymin = torch.min(abs(y), dim=1).values # minimum value for each 
                # unc_corrl, unc_sampling etc are considerd 
                sig = torch.Tensor( data.create_uncertainty(tuple(y.shape), expand=c.n_sig_MC ) ) # n_rows = B x n_sig_MC
                if c.n_sig_MC*c.n_noise_MC > 1:
                    # N_mc expansion (n_noise_MC) + noise sampling (all line independent)
                    x = torch.repeat_interleave(x, c.n_sig_MC*c.n_noise_MC, dim=0) 
                    y = torch.repeat_interleave(y, c.n_sig_MC*c.n_noise_MC, dim=0)
                    sig = torch.repeat_interleave(sig, c.n_noise_MC, dim=0)
                y = torch.clip( y * (1+ (10**(sig))*torch.randn(y.shape[0], y.shape[1]) ), min=ymin.reshape(-1,1) ) # all line independent
                
                # Transform to rescaled: np array again
                x = torch.Tensor(data.params_to_x(x)).to(c.device)
                y = torch.hstack((torch.Tensor(data.obs_to_y(y)), torch.Tensor(data.unc_to_sig(10**sig)))).to(c.device)  
                
                
                features = model.cond_net.features(y)
                
                output, jac = model.model(x, features)
    
                zz = torch.sum(output**2, dim=1)
    
                neg_log_likeli = 0.5 * zz - jac
    
                l = torch.mean(neg_log_likeli)
                l.backward(retain_graph=c.do_rev)
    
                if c.do_rev:
                    samples_noisy = sample_z(c.latent_noise) + output.data
                    # original code raised AssertionError (no condition input)
                    # x_rec = model.model(samples_noisy, rev=True)
                    x_rec, _ = model.model(samples_noisy, features, rev=True)
                    l_rev = torch.mean( (x-x_rec)**2 )
                    l_rev.backward()
                else:
                    l_rev = dummy_loss()
    
                model.optim_step()
                loss_history.append([l.item(), l_rev.item()])
    
                if i_batch+1 >= c.n_its_per_epoch:
                    # somehow the data loader workers don't shut down automatically
                    try:
                        data_iter._shutdown_workers()
                    except:
                        pass
    
                    break
        
            # Loop for test_loader, Get loss for test set
            for i_batch, test_tuple in tqdm.tqdm(enumerate(test_iter),
                                                  total=min(len(test_loader), c.n_its_per_epoch),
                                                  leave=False,
                                                  mininterval=1.,
                                                  disable=(not c.progress_bar),
                                                  ncols=83):
    
                x, y = test_tuple
                
                ymin = torch.min(abs(y), dim=1).values # minimum value for each 
                # unc_corrl, unc_sampling etc are considerd 
                sig = torch.Tensor( data.create_uncertainty(tuple(y.shape), expand=c.n_sig_MC ) ) # n_rows = B x n_sig_MC
                if c.n_sig_MC*c.n_noise_MC > 1:
                    # N_mc expansion (n_noise_MC) + noise sampling (all line independent)
                    x = torch.repeat_interleave(x, c.n_sig_MC*c.n_noise_MC, dim=0) 
                    y = torch.repeat_interleave(y, c.n_sig_MC*c.n_noise_MC, dim=0)
                    sig = torch.repeat_interleave(sig, c.n_noise_MC, dim=0)
                y = torch.clip( y * (1+ (10**(sig))*torch.randn(y.shape[0], y.shape[1]) ), min=ymin.reshape(-1,1) ) # all line independent
                
                # Transform to rescaled: np array again
                x = torch.Tensor(data.params_to_x(x)).to(c.device)
                y = torch.hstack((torch.Tensor(data.obs_to_y(y)), torch.Tensor(data.unc_to_sig(10**sig)))).to(c.device)  
    
                features = model.cond_net.features(y)
                with torch.no_grad():
                    output, jac = model.model(x, features)
                        
                zz = torch.sum(output**2, dim=1)
                neg_log_likeli = 0.5 * zz - jac
    
                tl = torch.mean(neg_log_likeli)
                # DO NOT BACKWARD FOR TEST!! USE NO_GRAD()
                # tl.backward(retain_graph=c.do_rev)
    
                if c.do_rev:
                    samples_noisy = sample_z(c.latent_noise) + output.data
                    with torch.no_grad():
                        x_rec, _ = model.model(samples_noisy, features, rev=True)
        
                        tl_rev = torch.mean( (x-x_rec)**2 )
                else:
                    tl_rev = dummy_loss()
                # tl_rev = dummy_loss()
    
                test_loss_history.append([tl.item(), tl_rev.item()])
    
                if i_batch+1 >= c.n_its_per_epoch:
                    # somehow the data loader workers don't shut down automatically
                    try:
                        test_iter._shutdown_workers()
                    except:
                        pass
    
                    break
                
            model.weight_scheduler.step()
            
            # calculate loss of this epoch
            epoch_losses_mn = np.mean(np.array(loss_history), axis=0) # 이걸 mean 하는지 median하는지에 따라서. 
            test_epoch_losses_mn = np.mean(np.array(test_loss_history), axis=0)
            
            epoch_losses_mdn = np.median(np.array(loss_history), axis=0)
            test_epoch_losses_mdn = np.median(np.array(test_loss_history), axis=0)
            
            epoch_loss_history['Loss_train_mn'][i_epoch] = epoch_losses_mn[0]
            epoch_loss_history['Loss_test_mn'][i_epoch] = test_epoch_losses_mn[0]
            epoch_loss_history['Loss_train_mdn'][i_epoch] = epoch_losses_mdn[0]
            epoch_loss_history['Loss_test_mdn'][i_epoch] = test_epoch_losses_mdn[0]
            
            epoch_loss_history['L_rev_train_mn'][i_epoch] = epoch_losses_mn[1]
            epoch_loss_history['L_rev_test_mn'][i_epoch] = test_epoch_losses_mn[1]
            epoch_loss_history['L_rev_train_mdn'][i_epoch] = epoch_losses_mdn[1]
            epoch_loss_history['L_rev_test_mdn'][i_epoch] = test_epoch_losses_mdn[1]
                                           
            
            if verbose:
                if i_epoch >= 0 :
                    viz.show_loss([epoch_loss_history[k_name][i_epoch] for k_name in c.loss_names], 
                                  logscale=False, its=min(len(test_loader), c.n_its_per_epoch) )
            
                    output_orig = output.cpu()
                    viz.show_hist(output_orig) # pass
    
            with torch.no_grad():
                samples = sample_z(1.)
                pass
            
            if (i_epoch > 10)*((i_epoch % 20)==0):
                # loss_array = np.array(epoch_loss_history)
                fig, ax = plot_loss_curve_2types(
                                        epoch_loss_history['Epoch'][:i_epoch+1],
                                        epoch_loss_history['Loss_train_mn'][:i_epoch+1],
                                        epoch_loss_history['Loss_test_mn'][:i_epoch+1],
                                        epoch_loss_history['Loss_train_mdn'][:i_epoch+1],
                                        epoch_loss_history['Loss_test_mdn'][:i_epoch+1],
                                        # loss_array[:,0], loss_array[:,1], loss_array[:,3], 
                                            c=c, 
                                          figname=c.filename+'_Loss_plot.pdf', # 이부분 hyperearch에선 바꿔줘야? 아님 나중에 폴더로 이동?
                                          yrange=c.loss_plot_yrange, xrange=c.loss_plot_xrange, 
                                          title=os.path.basename(c.config_file).replace('.py','') )
            
            model.model.zero_grad()
    
            if c.checkpoint_save:
                if (i_epoch % c.checkpoint_save_interval) == 0:
                    model.save(c.filename + '_checkpoint_%.4i' % (i_epoch * (1-c.checkpoint_save_overwrite)))
                    
            #####
            # Check convergence and divergence and change training_status
            #####
            if i_epoch > N_CONV_CHECK:
                flag_train_conv = train_tools.check_convergence(epoch_loss_history[name_check_train][:i_epoch+1], 
                                                   conv_cut=CONV_CUT, n_conv_check=N_CONV_CHECK)
                flag_test_conv = train_tools.check_convergence(epoch_loss_history[name_check_test][:i_epoch+1],
                                               conv_cut=CONV_CUT, n_conv_check=N_CONV_CHECK)
                if flag_train_conv * flag_test_conv:
                    if verbose:
                        print("Both train and test curves converged at: %d"%i_epoch)
                    training_status = 1 # 1=converged
                    
            if i_epoch > N_DIVG_CHECK:
                flag_train_divg = train_tools.check_divergence(epoch_loss_history[name_check_train][:i_epoch+1],
                                                   chunk_size=DIVG_CHUNK_SIZE, n_divg_check=N_DIVG_CHECK, divg_cri=DIVG_CRI)
                flag_test_divg = train_tools.check_divergence(epoch_loss_history[name_check_test][:i_epoch+1], 
                                                  chunk_size=DIVG_CHUNK_SIZE, n_divg_check=N_DIVG_CHECK, divg_cri=DIVG_CRI)
                if flag_test_divg==True or flag_train_divg==True:
                    if verbose:
                        print("Either train or test curve diverged at: %d"%i_epoch)
                    training_status = -1 # -1 = diverged
                    
            
        
        
        ### END LOOP    
        final_epoch = i_epoch + 1
        # remove unnecessary data in loss 
        epoch_loss_history = epoch_loss_history[:final_epoch]
        # update config with actural n_epochs used
        c.n_epochs = final_epoch
        train_tools.rewrite_config_element(c.config_file,  new_config_file=c.config_file, 
                                           param_to_change='n_epochs', value_to_change=c.n_epochs)
        
        if verbose:
            if c.live_visualization:
                viz.visualizer.viz.text('<h1 style="color:green">Done</h1>', win=running_box)
        model.save(c.filename)
        
        # loss_array = np.array(epoch_loss_history)
        fig, ax = plot_loss_curve_2types(
                                    epoch_loss_history['Epoch'][:i_epoch+1],
                                    epoch_loss_history['Loss_train_mn'][:i_epoch+1],
                                    epoch_loss_history['Loss_test_mn'][:i_epoch+1],
                                    epoch_loss_history['Loss_train_mdn'][:i_epoch+1],
                                    epoch_loss_history['Loss_test_mdn'][:i_epoch+1],
                                  c=c, 
                                  figname=c.filename+'_Loss_plot.pdf', #여기도 주의
                                  yrange=c.loss_plot_yrange, xrange=c.loss_plot_xrange, 
                                  title=os.path.basename(c.config_file).replace('.py',''))
        
        # header = 'Epoch\tL_train\tlr_train\tL_test\tlr_test'
        header = '\t'.join(epoch_loss_history.dtype.names)
        formats = []
        for name in epoch_loss_history.dtype.names:
            if epoch_loss_history[name].dtype == np.int32:
                formats.append('%d')
            else:
                formats.append('%.8f')
         
        np.savetxt(c.filename+'_loss_history.txt', epoch_loss_history, delimiter='\t', fmt='\t'.join(formats), header=header)

    except:
        model.save(c.filename + '_ABORT')
        # loss_array = np.array(epoch_loss_history)
        if len(epoch_loss_history) > 10:
            fig, ax = plot_loss_curve_2types(
                                    epoch_loss_history['Epoch'][:i_epoch+1],
                                    epoch_loss_history['Loss_train_mn'][:i_epoch+1],
                                    epoch_loss_history['Loss_test_mn'][:i_epoch+1],
                                    epoch_loss_history['Loss_train_mdn'][:i_epoch+1],
                                    epoch_loss_history['Loss_test_mdn'][:i_epoch+1],
                                    c=c, 
                                      figname=c.filename+'_ABORT_Loss_plot.pdf',#여기도 주의
                                     title=os.path.basename(c.config_file).replace('.py',''))
        raise
    
    t_end = time()
    print('Time taken for training: {:.1f} hour ({:.1f} min)'.format( (t_end-t_start)/3600, (t_end-t_start)/60 ) )

    
"""
Training code for SimGap network : FTransformNet
"""
def train_ftrans_network(c, feature_test=None, feature_train=None, 
                  verbose=True, max_epoch=200, return_model=False, return_training_status=False):
   
    t_start = time()
    
    from .tools import simgap_tools
    
    def sample_z(sigma):
    #    return sigma * torch.cuda.FloatTensor(c.batch_size, c.x_dim).normal_()
        return sigma * torch.FloatTensor(c.batch_size, c.x_dim).normal_().to(c.device)
    
    # make savedir
    if (not os.path.exists( os.path.dirname(c.filename) )) and ( os.path.dirname(c.filename)!= '' ):
        os.system('mkdir -p '+os.path.dirname(c.filename))
        
    model = eval(c.model_code)(c)
    
    # make viz
    if verbose:
        viz = TrainVisualizer(c)
        if c.live_visualization:
            viz.visualizer.viz.text('<pre>' + config_str + '</pre>')
            running_box = viz.visualizer.viz.text('<h1 style="color:red">Running</h1>')
    
    try:
        epoch_loss_history = []
        
        # 저장해야할 것. 저장은 항상하고 그외에 다루는 걸 config가 조절해야
        # Loss_train_mn, Loss_test_mn, Loss_train_mdn, Loss_test_mdn
        # L_rev_train_mn, L_rev_test_mn, L_rev_train_mdn, L_rev_test_mdn
        loss_header = ['Epoch']
        loss_header +=['Loss_train_mn', 'Loss_test_mn', 'Loss_train_mdn', 'Loss_test_mdn']

        loss_format = ['int32'] + ['float32']*len(loss_header)
        epoch_loss_history = np.zeros(max_epoch, 
                                          dtype={'names':loss_header, 'formats':loss_format} )
        epoch_loss_history['Epoch'] = np.arange(max_epoch)
        
        ########
        # training status: -1, 0 or 1
        # 0: training in progress (not converged, not diverged)
        # 1: training converged (both train set and test set converged)
        # -1: training diverged (either train set or test set diverged)
        training_status = 0
        i_epoch = - 1
        
        name_check_train = 'Loss_train_mdn'
        name_check_test = 'Loss_test_mdn'
        
        while (training_status == 0 and i_epoch < max_epoch-1):
            i_epoch += 1
            
            test_loader = torch.utils.data.DataLoader( 
                        torch.utils.data.TensorDataset(torch.Tensor(feature_test)),
                        batch_size=c.batch_size, shuffle=True, drop_last=True)
                     
            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.Tensor(feature_train)),
                batch_size=c.batch_size, shuffle=True, drop_last=True)
            

            data_iter = iter(train_loader)
            test_iter = iter(test_loader)
    
            loss_history = [] # save loss for all batches
            test_loss_history = []
            
            if i_epoch < 0:
                for param_group in model.optim.param_groups:
                    param_group['lr'] = c.lr_init * 2e-2
                    
                    
            # Loop for train_loader, Optimization
            for i_batch, data_tuple in tqdm.tqdm(enumerate(data_iter),
                                                  total=min(len(train_loader), c.n_its_per_epoch),
                                                  leave=False,
                                                  mininterval=1.,
                                                  disable=(not c.progress_bar),
                                                  ncols=83):
    
                x = data_tuple[0]
                x = x.to(c.device)
                
                output, jac = model.model(x)  
                # output = model.model(x) 
                zz = torch.sum(output**2, dim=1)
    
                neg_log_likeli = 0.5 * zz - jac
    
                l = torch.mean(neg_log_likeli)
                l.backward(retain_graph=c.do_rev)
                l_rev = dummy_loss()
                
    
                model.optim_step()
                loss_history.append([l.item(), l_rev.item()])
                
    
                if i_batch+1 >= c.n_its_per_epoch:
                    # somehow the data loader workers don't shut down automatically
                    try:
                        data_iter._shutdown_workers()
                    except:
                        pass
    
                    break
            # Loop for test_loader, Get loss for test set
            for i_batch, test_tuple in tqdm.tqdm(enumerate(test_iter),
                                                  total=min(len(test_loader), c.n_its_per_epoch),
                                                  leave=False,
                                                  mininterval=1.,
                                                  disable=(not c.progress_bar),
                                                  ncols=83):
    
                x = test_tuple[0]
                x = x.to(c.device)
    
                with torch.no_grad():
                    output, jac = model.model(x)
                    # output = model.model(x) 
                zz = torch.sum(output**2, dim=1)
                neg_log_likeli = 0.5 * zz - jac
                
                tl = torch.mean(neg_log_likeli)
                # DO NOT BACKWARD FOR TEST!! USE NO_GRAD()
                # tl.backward(retain_graph=c.do_rev)
                
                tl_rev = dummy_loss()
            
                test_loss_history.append([tl.item(), tl_rev.item()])
            
    
                if i_batch+1 >= c.n_its_per_epoch:
                    # somehow the data loader workers don't shut down automatically
                    try:
                        test_iter._shutdown_workers()
                    except:
                        pass
                    break
            model.weight_scheduler.step()
            # calculate loss of this epoch
            epoch_losses_mn = np.mean(np.array(loss_history), axis=0) # 이걸 mean 하는지 median하는지에 따라서. 
            test_epoch_losses_mn = np.mean(np.array(test_loss_history), axis=0)
            
            epoch_losses_mdn = np.median(np.array(loss_history), axis=0)
            test_epoch_losses_mdn = np.median(np.array(test_loss_history), axis=0)
            
            epoch_loss_history['Loss_train_mn'][i_epoch] = epoch_losses_mn[0]
            epoch_loss_history['Loss_test_mn'][i_epoch] = test_epoch_losses_mn[0]
            epoch_loss_history['Loss_train_mdn'][i_epoch] = epoch_losses_mdn[0]
            epoch_loss_history['Loss_test_mdn'][i_epoch] = test_epoch_losses_mdn[0]
            
            if verbose:
                if i_epoch >=0:
                    viz.show_loss([epoch_loss_history[k_name][i_epoch] for k_name in c.loss_names], 
                                  logscale=False, its=min(len(test_loader), c.n_its_per_epoch) )
                    output_orig = output.cpu()
                    viz.show_hist(output_orig) # pass
                    
            if (i_epoch > 10)*((i_epoch % 20)==0):
                # loss_array = np.array(epoch_loss_history)
                fig, ax = plot_loss_curve_2types(
                                        epoch_loss_history['Epoch'][:i_epoch+1],
                                        epoch_loss_history['Loss_train_mn'][:i_epoch+1],
                                        epoch_loss_history['Loss_test_mn'][:i_epoch+1],
                                        epoch_loss_history['Loss_train_mdn'][:i_epoch+1],
                                        epoch_loss_history['Loss_test_mdn'][:i_epoch+1],
                                        # loss_array[:,0], loss_array[:,1], loss_array[:,3], 
                                            c=c, 
                                          figname=c.filename+'_Loss_plot.pdf', # 이부분 hyperearch에선 바꿔줘야? 아님 나중에 폴더로 이동?
                                          # yrange=c.loss_plot_yrange, xrange=c.loss_plot_xrange, 
                                           title=os.path.basename(c.config_file).replace('.py',''))
                                          # title=os.path.basename(c.config_file).replace('.py','') )
                
            
            model.model.zero_grad()
            
            #####
            # Check convergence and divergence and change training_status
            #####
            if i_epoch > N_CONV_CHECK:
                flag_train_conv = train_tools.check_convergence(epoch_loss_history[name_check_train][:i_epoch+1], 
                                                   conv_cut=CONV_CUT, n_conv_check=N_CONV_CHECK)
                flag_test_conv = train_tools.check_convergence(epoch_loss_history[name_check_test][:i_epoch+1],
                                               conv_cut=CONV_CUT, n_conv_check=N_CONV_CHECK)
                if flag_train_conv * flag_test_conv:
                    if verbose:
                        print("Both train and test curves converged at: %d"%i_epoch)
                    training_status = 1 # 1=converged
                    
            if i_epoch > N_DIVG_CHECK:
                flag_train_divg = train_tools.check_divergence(epoch_loss_history[name_check_train][:i_epoch+1],
                                                   chunk_size=DIVG_CHUNK_SIZE, n_divg_check=N_DIVG_CHECK, divg_cri=DIVG_CRI)
                flag_test_divg = train_tools.check_divergence(epoch_loss_history[name_check_test][:i_epoch+1], 
                                                  chunk_size=DIVG_CHUNK_SIZE, n_divg_check=N_DIVG_CHECK, divg_cri=DIVG_CRI)
                if flag_test_divg==True or flag_train_divg==True:
                    if verbose:
                        print("Either train or test curve diverged at: %d"%i_epoch)
                    training_status = -1 # -1 = diverged
                    
        ### END LOOP    
        final_epoch = i_epoch + 1
        # remove unnecessary data in loss 
        epoch_loss_history = epoch_loss_history[:final_epoch]
        # update config with actural n_epochs used
        c.n_epochs = final_epoch
        # train_tools.rewrite_config_element(c.config_file,  new_config_file=c.config_file, 
        #                                    param_to_change='n_epochs', value_to_change=c.n_epochs)
        
        model.save(c.filename)
        if verbose:
            if c.live_visualization:
                viz.visualizer.viz.text('<h1 style="color:green">Done</h1>', win=running_box)
        fig, ax = plot_loss_curve_2types(
                                    epoch_loss_history['Epoch'][:i_epoch+1],
                                    epoch_loss_history['Loss_train_mn'][:i_epoch+1],
                                    epoch_loss_history['Loss_test_mn'][:i_epoch+1],
                                    epoch_loss_history['Loss_train_mdn'][:i_epoch+1],
                                    epoch_loss_history['Loss_test_mdn'][:i_epoch+1],
                                    # loss_array[:,0], loss_array[:,1], loss_array[:,3], 
                                        c=c, 
                                      figname=c.filename+'_Loss_plot.pdf', # 이부분 hyperearch에선 바꿔줘야? 아님 나중에 폴더로 이동?
                                      # yrange=c.loss_plot_yrange, xrange=c.loss_plot_xrange, 
                                       title=os.path.basename(c.config_file).replace('.py',''))
       
        
        # header = 'Epoch\tL_train\tlr_train\tL_test\tlr_test'
        header = '\t'.join(epoch_loss_history.dtype.names)
        formats = []
        for name in epoch_loss_history.dtype.names:
            if epoch_loss_history[name].dtype == np.int32:
                formats.append('%d')
            else:
                formats.append('%.8f')
         
        np.savetxt(c.filename+'_loss_history.txt', epoch_loss_history, delimiter='\t', fmt='\t'.join(formats), header=header)
        
        
        return_output=[]
        if return_model:
            return_output.append(model)
        if return_training_status:
            return_output.append(training_status)
        if len(return_output)>1:
            return return_output
    
    except:
        # model.save(c.filename + '_ABORT')
        # loss_array = np.array(epoch_loss_history)
        if len(epoch_loss_history) > 10:
            fig, ax = plot_loss_curve_2types(
                                    epoch_loss_history['Epoch'][:i_epoch+1],
                                    epoch_loss_history['Loss_train_mn'][:i_epoch+1],
                                    epoch_loss_history['Loss_test_mn'][:i_epoch+1],
                                    epoch_loss_history['Loss_train_mdn'][:i_epoch+1],
                                    epoch_loss_history['Loss_test_mdn'][:i_epoch+1],
                                    c=c, 
                                      figname=c.filename+'_ABORT_Loss_plot.pdf',#여기도 주의
                                    # title='Test')
                                     title=os.path.basename(c.config_file).replace('.py',''))
        raise
    
    t_end = time()
    print('Time taken for training: {:.1f} hour ({:.1f} min)'.format( (t_end-t_start)/3600, (t_end-t_start)/60 ) )
    
    
"""
Train Normal-Net with flag
"""
def train_flag_network(c, data=None, verbose=True, max_epoch=1000): # c is cINNConfig class variable

    t_start = time()
    
    # print all parameters in the config
    if verbose:
        config_str = ""
        config_str += "==="*30 + "\n"
        config_str += "Config options:\n\n"

        print_list = c.parameter_list
        if len(c.x_names) > 20: print_list.remove('x_names')
        if len(c.y_names) > 20: print_list.remove('y_names')
        for key in c.flag_names:
            if len(c.flag_dic[key]) > 20: 
                print_list.remove("flag_dic")
                print_list.remove("flag_index_dic")
                break

        for param in print_list:
            if getattr(c, param) is not None:
                config_str += "  {:25}\t{}\n".format(param, getattr(c, param))

        config_str += "==="*30 + "\n"
        print(config_str)

    model = eval(c.model_code)(c)
    
    # make savedir
    if (not os.path.exists( os.path.dirname(c.filename) )) and ( os.path.dirname(c.filename)!= '' ):
        os.system('mkdir -p '+os.path.dirname(c.filename))
        
    # make viz
    if verbose:
        viz = TrainVisualizer(c)
        if c.live_visualization:
            viz.visualizer.viz.text('<pre>' + config_str + '</pre>')
            running_box = viz.visualizer.viz.text('<h1 style="color:red">Running</h1>')
           
        
    if c.load_file:
        model.load(c.load_file, device=c.device)
        
        
    def sample_z(sigma):
    #    return sigma * torch.cuda.FloatTensor(c.batch_size, c.x_dim).normal_()
        return sigma * torch.FloatTensor(c.batch_size, c.x_dim).normal_().to(c.device)
    
    try:
        epoch_loss_history = []
        
        # 저장해야할 것. 저장은 항상하고 그외에 다루는 걸 config가 조절해야
        # Loss_train_mn, Loss_test_mn, Loss_train_mdn, Loss_test_mdn
        # L_rev_train_mn, L_rev_test_mn, L_rev_train_mdn, L_rev_test_mdn
        loss_header = ['Epoch']
        loss_header +=['Loss_train_mn', 'Loss_test_mn', 'Loss_train_mdn', 'Loss_test_mdn']
        # if c.do_rev:
        loss_header += ['L_rev_train_mn', 'L_rev_test_mn', 'L_rev_train_mdn', 'L_rev_test_mdn']

        loss_format = ['int32'] + ['float32']*len(loss_header)
        epoch_loss_history = np.zeros(max_epoch, 
                                          dtype={'names':loss_header, 'formats':loss_format} )
        epoch_loss_history['Epoch'] = np.arange(max_epoch)
        
        if data is None:
            data = DataLoader(c, update_rescale_parameters=True)
        
        if model.rescale_params is None:
            # pass rescale parameter information
#             model.rescale_params = {'mu_x': data.mu_x, 'mu_y':data.mu_y, 'w_x':data.w_x, 'w_y':data.w_y}
            model.rescale_params = {'mu_x': data.mu_x, 'mu_y':data.mu_y, 'w_x':data.w_x, 'w_y':data.w_y, 'mu_f': data.mu_f, 'w_f': data.w_f}
            
        
        ########
        # training status: -1, 0 or 1
        # 0: training in progress (not converged, not diverged)
        # 1: training converged (both train set and test set converged)
        # -1: training diverged (either train set or test set diverged)
        training_status = 0
        i_epoch = - 1
        
        name_check_train = 'Loss_train_mdn'
        name_check_test = 'Loss_test_mdn'
        
        while (training_status == 0 and i_epoch < max_epoch-1):
            i_epoch += 1
        # for i_epoch in range(-c.pre_low_lr, c.n_epochs):
            
            test_loader, train_loader = data.get_loaders(param_seed = i_epoch) # param_seed will randomize parameter if random_parameters is set
        
            data_iter = iter(train_loader)
            test_iter = iter(test_loader)
    
            loss_history = [] # save loss for all batches
            test_loss_history = []
            
            if i_epoch < 0:
                for param_group in model.optim.param_groups:
                    param_group['lr'] = c.lr_init * 2e-2
                    
            # Loop for train_loader, Optimization
            for i_batch, data_tuple in tqdm.tqdm(enumerate(data_iter),
                                                  total=min(len(train_loader), c.n_its_per_epoch),
                                                  leave=False,
                                                  mininterval=1.,
                                                  disable=(not c.progress_bar),
                                                  ncols=83):
                
                
                
                # This is for Normal-Net, so x y are rescaled values, but random flag is not rescaled yet
                x, y = data_tuple
                
        
                flags = data.create_random_flag( N_data = y.shape[0]  ) # numpy array
    
                # turn off if flag=0
                for i_flag, flag_name in enumerate(c.flag_names):
                    roi_off = np.where(flags[:, i_flag] == 0.0)[0]
                    y[roi_off][:,c.flag_index_dic[flag_name]] = 0.0
                
                
                # transform flags to rf and hstack with y
                y = torch.hstack( (y.to(c.device),  torch.Tensor(data.flag_to_rf(flags)).to(c.device) ) ).to(c.device)  
                x = x.to(c.device)
                
                features = model.cond_net.features(y)
                
                output, jac = model.model(x, features)
                
                zz = torch.sum(output**2, dim=1)
    
                neg_log_likeli = 0.5 * zz - jac
    
                l = torch.mean(neg_log_likeli)
                l.backward(retain_graph=c.do_rev)
    
                if c.do_rev:
                    samples_noisy = sample_z(c.latent_noise) + output.data
                    # original code raised AssertionError (no condition input)
                    # x_rec = model.model(samples_noisy, rev=True)
                    x_rec, _ = model.model(samples_noisy, features, rev=True)
                    
                    l_rev = torch.mean( (x-x_rec)**2 )
                    l_rev.backward()
                else:
                    l_rev = dummy_loss()
    
                model.optim_step()
                loss_history.append([l.item(), l_rev.item()])
    
                if i_batch+1 >= c.n_its_per_epoch:
                    # somehow the data loader workers don't shut down automatically
                    try:
                        data_iter._shutdown_workers()
                    except:
                        pass
    
                    break
        
            # Loop for test_loader, Get loss for test set
            for i_batch, test_tuple in tqdm.tqdm(enumerate(test_iter),
                                                  total=min(len(test_loader), c.n_its_per_epoch),
                                                  leave=False,
                                                  mininterval=1.,
                                                  disable=(not c.progress_bar),
                                                  ncols=83):
    
                x, y = test_tuple
                
                flags = data.create_random_flag( N_data = y.shape[0]  ) # numpy array
    
                # turn off if flag=0
                for i_flag, flag_name in enumerate(c.flag_names):
                    roi_off = np.where(flags[:, i_flag] == 0.0)[0]
                    y[roi_off][:,c.flag_index_dic[flag_name]] = 0.0
                
                
                # transform flags to rf and hstack with y
                y = torch.hstack( (y.to(c.device),  torch.Tensor(data.flag_to_rf(flags)).to(c.device) ) ).to(c.device)  
                x = x.to(c.device)
                
                features = model.cond_net.features(y)
                with torch.no_grad():
                    output, jac = model.model(x, features)
                    
                zz = torch.sum(output**2, dim=1)
                neg_log_likeli = 0.5 * zz - jac
    
                tl = torch.mean(neg_log_likeli)
                # DO NOT BACKWARD FOR TEST!! USE NO_GRAD()
                # tl.backward(retain_graph=c.do_rev)
    
                if c.do_rev:
                    samples_noisy = sample_z(c.latent_noise) + output.data
                    with torch.no_grad():
                        x_rec, _ = model.model(samples_noisy, features, rev=True)
                        
                        tl_rev = torch.mean( (x-x_rec)**2 )
                else:
                    tl_rev = dummy_loss()
                # tl_rev = dummy_loss()
    
                test_loss_history.append([tl.item(), tl_rev.item()])
    
                if i_batch+1 >= c.n_its_per_epoch:
                    # somehow the data loader workers don't shut down automatically
                    try:
                        test_iter._shutdown_workers()
                    except:
                        pass
    
                    break
                
            model.weight_scheduler.step()
            
            # calculate loss of this epoch
            epoch_losses_mn = np.mean(np.array(loss_history), axis=0) # 이걸 mean 하는지 median하는지에 따라서. 
            test_epoch_losses_mn = np.mean(np.array(test_loss_history), axis=0)
            
            epoch_losses_mdn = np.median(np.array(loss_history), axis=0)
            test_epoch_losses_mdn = np.median(np.array(test_loss_history), axis=0)
            
            epoch_loss_history['Loss_train_mn'][i_epoch] = epoch_losses_mn[0]
            epoch_loss_history['Loss_test_mn'][i_epoch] = test_epoch_losses_mn[0]
            epoch_loss_history['Loss_train_mdn'][i_epoch] = epoch_losses_mdn[0]
            epoch_loss_history['Loss_test_mdn'][i_epoch] = test_epoch_losses_mdn[0]
            
            epoch_loss_history['L_rev_train_mn'][i_epoch] = epoch_losses_mn[1]
            epoch_loss_history['L_rev_test_mn'][i_epoch] = test_epoch_losses_mn[1]
            epoch_loss_history['L_rev_train_mdn'][i_epoch] = epoch_losses_mdn[1]
            epoch_loss_history['L_rev_test_mdn'][i_epoch] = test_epoch_losses_mdn[1]
                                           
            
            if verbose:
                if i_epoch >= 0:
                    viz.show_loss([epoch_loss_history[k_name][i_epoch] for k_name in c.loss_names], 
                                  logscale=False, its=min(len(test_loader), c.n_its_per_epoch) )
            
                    output_orig = output.cpu()
                    viz.show_hist(output_orig) # pass
    
            with torch.no_grad():
                samples = sample_z(1.)
                pass
            
            if (i_epoch > 10)*((i_epoch % 20)==0):
                # loss_array = np.array(epoch_loss_history)
                fig, ax = plot_loss_curve_2types(
                                        epoch_loss_history['Epoch'][:i_epoch+1],
                                        epoch_loss_history['Loss_train_mn'][:i_epoch+1],
                                        epoch_loss_history['Loss_test_mn'][:i_epoch+1],
                                        epoch_loss_history['Loss_train_mdn'][:i_epoch+1],
                                        epoch_loss_history['Loss_test_mdn'][:i_epoch+1],
                                        # loss_array[:,0], loss_array[:,1], loss_array[:,3], 
                                            c=c, 
                                          figname=c.filename+'_Loss_plot.pdf', # 이부분 hyperearch에선 바꿔줘야? 아님 나중에 폴더로 이동?
                                          yrange=c.loss_plot_yrange, xrange=c.loss_plot_xrange, 
                                          title=os.path.basename(c.config_file).replace('.py','') )
            
            model.model.zero_grad()
    
            if c.checkpoint_save:
                if (i_epoch % c.checkpoint_save_interval) == 0:
                    model.save(c.filename + '_checkpoint_%.4i' % (i_epoch * (1-c.checkpoint_save_overwrite)))
                    
            #####
            # Check convergence and divergence and change training_status
            #####
            if i_epoch > N_CONV_CHECK:
                flag_train_conv = train_tools.check_convergence(epoch_loss_history[name_check_train][:i_epoch+1], 
                                                   conv_cut=CONV_CUT, n_conv_check=N_CONV_CHECK)
                flag_test_conv = train_tools.check_convergence(epoch_loss_history[name_check_test][:i_epoch+1],
                                               conv_cut=CONV_CUT, n_conv_check=N_CONV_CHECK)
                if flag_train_conv * flag_test_conv:
                    if verbose:
                        print("Both train and test curves converged at: %d"%i_epoch)
                    training_status = 1 # 1=converged
                    
            if i_epoch > N_DIVG_CHECK:
                flag_train_divg = train_tools.check_divergence(epoch_loss_history[name_check_train][:i_epoch+1],
                                                   chunk_size=DIVG_CHUNK_SIZE, n_divg_check=N_DIVG_CHECK, divg_cri=DIVG_CRI)
                flag_test_divg = train_tools.check_divergence(epoch_loss_history[name_check_test][:i_epoch+1], 
                                                  chunk_size=DIVG_CHUNK_SIZE, n_divg_check=N_DIVG_CHECK, divg_cri=DIVG_CRI)
                if flag_test_divg==True or flag_train_divg==True:
                    if verbose:
                        print("Either train or test curve diverged at: %d"%i_epoch)
                    training_status = -1 # -1 = diverged
                    
            
        
        
        ### END LOOP    
        final_epoch = i_epoch + 1
        # remove unnecessary data in loss 
        epoch_loss_history = epoch_loss_history[:final_epoch]
        # update config with actural n_epochs used
        c.n_epochs = final_epoch
        train_tools.rewrite_config_element(c.config_file,  new_config_file=c.config_file, 
                                           param_to_change='n_epochs', value_to_change=c.n_epochs)
        
        if verbose:
            if c.live_visualization:
                viz.visualizer.viz.text('<h1 style="color:green">Done</h1>', win=running_box)
        model.save(c.filename)
        
        # loss_array = np.array(epoch_loss_history)
        fig, ax = plot_loss_curve_2types(
                                    epoch_loss_history['Epoch'][:i_epoch+1],
                                    epoch_loss_history['Loss_train_mn'][:i_epoch+1],
                                    epoch_loss_history['Loss_test_mn'][:i_epoch+1],
                                    epoch_loss_history['Loss_train_mdn'][:i_epoch+1],
                                    epoch_loss_history['Loss_test_mdn'][:i_epoch+1],
                                  c=c, 
                                  figname=c.filename+'_Loss_plot.pdf', #여기도 주의
                                  yrange=c.loss_plot_yrange, xrange=c.loss_plot_xrange, 
                                  title=os.path.basename(c.config_file).replace('.py',''))
        
        # header = 'Epoch\tL_train\tlr_train\tL_test\tlr_test'
        header = '\t'.join(epoch_loss_history.dtype.names)
        formats = []
        for name in epoch_loss_history.dtype.names:
            if epoch_loss_history[name].dtype == np.int32:
                formats.append('%d')
            else:
                formats.append('%.8f')
         
        np.savetxt(c.filename+'_loss_history.txt', epoch_loss_history, delimiter='\t', fmt='\t'.join(formats), header=header)

    except:
        model.save(c.filename + '_ABORT')
        # loss_array = np.array(epoch_loss_history)
        if len(epoch_loss_history) > 10:
            fig, ax = plot_loss_curve_2types(
                                    epoch_loss_history['Epoch'][:i_epoch+1],
                                    epoch_loss_history['Loss_train_mn'][:i_epoch+1],
                                    epoch_loss_history['Loss_test_mn'][:i_epoch+1],
                                    epoch_loss_history['Loss_train_mdn'][:i_epoch+1],
                                    epoch_loss_history['Loss_test_mdn'][:i_epoch+1],
                                    c=c, 
                                      figname=c.filename+'_ABORT_Loss_plot.pdf',#여기도 주의
                                     title=os.path.basename(c.config_file).replace('.py',''))
        raise
    
    t_end = time()
    print('Time taken for training: {:.1f} hour ({:.1f} min)'.format( (t_end-t_start)/3600, (t_end-t_start)/60 ) )

    


"""
Train normal net with wavelength coupling
"""

def train_wc_network(c, data=None, verbose=True, max_epoch=1000): # c is cINNConfig class variable

    t_start = time()

    # print all parameters in the config
    if verbose:
        config_str = ""
        config_str += "==="*30 + "\n"
        config_str += "Config options:\n\n"

        print_list = c.parameter_list
        if len(c.x_names) > 20: print_list.remove('x_names')
        if len(c.y_names) > 20: print_list.remove('y_names')
        
        for param in print_list:
            if getattr(c, param) is not None:
                config_str += "  {:25}\t{}\n".format(param, getattr(c, param))

        config_str += "==="*30 + "\n"
        print(config_str)
  
    model = eval(c.model_code)(c)
    
    # make savedir
    if (not os.path.exists( os.path.dirname(c.filename) )) and ( os.path.dirname(c.filename)!= '' ):
        os.system('mkdir -p '+os.path.dirname(c.filename))
        
    # make viz
    if verbose:
        viz = TrainVisualizer(c)
        if c.live_visualization:
            viz.visualizer.viz.text('<pre>' + config_str + '</pre>')
            running_box = viz.visualizer.viz.text('<h1 style="color:red">Running</h1>')
           
        
    if c.load_file:
        model.load(c.load_file, device=c.device)
        
        
    def sample_z(sigma):
    #    return sigma * torch.cuda.FloatTensor(c.batch_size, c.x_dim).normal_()
        return sigma * torch.FloatTensor(c.batch_size, c.x_dim).normal_().to(c.device)
    
    try:
        epoch_loss_history = []
        
        # 저장해야할 것. 저장은 항상하고 그외에 다루는 걸 config가 조절해야
        # Loss_train_mn, Loss_test_mn, Loss_train_mdn, Loss_test_mdn
        # L_rev_train_mn, L_rev_test_mn, L_rev_train_mdn, L_rev_test_mdn
        loss_header = ['Epoch']
        loss_header +=['Loss_train_mn', 'Loss_test_mn', 'Loss_train_mdn', 'Loss_test_mdn']
        # if c.do_rev:
        loss_header += ['L_rev_train_mn', 'L_rev_test_mn', 'L_rev_train_mdn', 'L_rev_test_mdn']

        loss_format = ['int32'] + ['float32']*len(loss_header)
        epoch_loss_history = np.zeros(max_epoch, 
                                          dtype={'names':loss_header, 'formats':loss_format} )
        epoch_loss_history['Epoch'] = np.arange(max_epoch)
        
        if data is None:
            data = DataLoader(c, update_rescale_parameters=True)
        
        if model.rescale_params is None:
            # pass rescale parameter information
#             model.rescale_params = {'mu_x': data.mu_x, 'mu_y':data.mu_y, 'w_x':data.w_x, 'w_y':data.w_y}
            model.rescale_params = {'mu_x': data.mu_x, 'mu_y':data.mu_y, 'w_x':data.w_x, 'w_y':data.w_y, 'mu_wl': data.mu_wl, 'w_wl': data.w_wl}
            
        
        ########
        # training status: -1, 0 or 1
        # 0: training in progress (not converged, not diverged)
        # 1: training converged (both train set and test set converged)
        # -1: training diverged (either train set or test set diverged)
        training_status = 0
        i_epoch = - 1
        
        name_check_train = 'Loss_train_mdn'
        name_check_test = 'Loss_test_mdn'
        
        while (training_status == 0 and i_epoch < max_epoch-1):
            i_epoch += 1
            # rawval=False
            test_loader, train_loader = data.get_loaders(param_seed = i_epoch) # param_seed will randomize parameter if random_parameters is set
        
            data_iter = iter(train_loader)
            test_iter = iter(test_loader)
    
            loss_history = [] # save loss for all batches
            test_loss_history = []
            
            if i_epoch < 0:
                for param_group in model.optim.param_groups:
                    param_group['lr'] = c.lr_init * 2e-2
                    
            # Loop for train_loader, Optimization
            for i_batch, data_tuple in tqdm.tqdm(enumerate(data_iter),
                                                  total=min(len(train_loader), c.n_its_per_epoch),
                                                  leave=False,
                                                  mininterval=1.,
                                                  disable=(not c.progress_bar),
                                                  ncols=83):
                
                # In wavelength coupling mode (wavelength_coupling=True), data are normalized, torch Tensor
                # Only thing to do is add wavelength -> permute Y and WL in the same order -> transform wl to lambda -> make torch Tensor -> attach y and lambda
                x, y = data_tuple
                
                wl = data.create_coupling_wavelength(y.shape[0]) 
                # Transform to rescaled: wl - lambda & torch Tensor
                lam = torch.Tensor(data.wl_to_lambda(wl))
                # permute (flux and wl together)
                perm = torch.randperm(y.shape[1])
                y = torch.hstack( (y[:,perm], lam[:,perm]) )
                # device
                x, y = x.to(c.device), y.to(c.device)
                
                
                features = model.cond_net.features(y)
                
                output, jac = model.model(x, features)
    
                zz = torch.sum(output**2, dim=1)
    
                neg_log_likeli = 0.5 * zz - jac
    
                l = torch.mean(neg_log_likeli)
                l.backward(retain_graph=c.do_rev)
    
                if c.do_rev:
                    samples_noisy = sample_z(c.latent_noise) + output.data
                    # original code raised AssertionError (no condition input)
                    # x_rec = model.model(samples_noisy, rev=True)
                    x_rec, _ = model.model(samples_noisy, features, rev=True)
                    l_rev = torch.mean( (x-x_rec)**2 )
                    l_rev.backward()
                else:
                    l_rev = dummy_loss()
    
                model.optim_step()
                loss_history.append([l.item(), l_rev.item()])
    
                if i_batch+1 >= c.n_its_per_epoch:
                    # somehow the data loader workers don't shut down automatically
                    try:
                        data_iter._shutdown_workers()
                    except:
                        pass
    
                    break
        
            # Loop for test_loader, Get loss for test set
            for i_batch, test_tuple in tqdm.tqdm(enumerate(test_iter),
                                                  total=min(len(test_loader), c.n_its_per_epoch),
                                                  leave=False,
                                                  mininterval=1.,
                                                  disable=(not c.progress_bar),
                                                  ncols=83):
    
                x, y = test_tuple
                
                wl = data.create_coupling_wavelength(y.shape[0]) 
                # Transform to rescaled: wl - lambda & torch Tensor
                lam = torch.Tensor(data.wl_to_lambda(wl))
                # permute (flux and wl together)
                perm = torch.randperm(y.shape[1])
                y = torch.hstack( (y[:,perm], lam[:,perm]) )
                # device
                x, y = x.to(c.device), y.to(c.device)  
    
                features = model.cond_net.features(y)
                with torch.no_grad():
                    output, jac = model.model(x, features)
                        
                zz = torch.sum(output**2, dim=1)
                neg_log_likeli = 0.5 * zz - jac
    
                tl = torch.mean(neg_log_likeli)
                # DO NOT BACKWARD FOR TEST!! USE NO_GRAD()
                # tl.backward(retain_graph=c.do_rev)
    
                if c.do_rev:
                    samples_noisy = sample_z(c.latent_noise) + output.data
                    with torch.no_grad():
                        x_rec, _ = model.model(samples_noisy, features, rev=True)
        
                        tl_rev = torch.mean( (x-x_rec)**2 )
                else:
                    tl_rev = dummy_loss()
                # tl_rev = dummy_loss()
    
                test_loss_history.append([tl.item(), tl_rev.item()])
    
                if i_batch+1 >= c.n_its_per_epoch:
                    # somehow the data loader workers don't shut down automatically
                    try:
                        test_iter._shutdown_workers()
                    except:
                        pass
    
                    break
                
            model.weight_scheduler.step()
            
            # calculate loss of this epoch
            epoch_losses_mn = np.mean(np.array(loss_history), axis=0) # 이걸 mean 하는지 median하는지에 따라서. 
            test_epoch_losses_mn = np.mean(np.array(test_loss_history), axis=0)
            
            epoch_losses_mdn = np.median(np.array(loss_history), axis=0)
            test_epoch_losses_mdn = np.median(np.array(test_loss_history), axis=0)
            
            epoch_loss_history['Loss_train_mn'][i_epoch] = epoch_losses_mn[0]
            epoch_loss_history['Loss_test_mn'][i_epoch] = test_epoch_losses_mn[0]
            epoch_loss_history['Loss_train_mdn'][i_epoch] = epoch_losses_mdn[0]
            epoch_loss_history['Loss_test_mdn'][i_epoch] = test_epoch_losses_mdn[0]
            
            epoch_loss_history['L_rev_train_mn'][i_epoch] = epoch_losses_mn[1]
            epoch_loss_history['L_rev_test_mn'][i_epoch] = test_epoch_losses_mn[1]
            epoch_loss_history['L_rev_train_mdn'][i_epoch] = epoch_losses_mdn[1]
            epoch_loss_history['L_rev_test_mdn'][i_epoch] = test_epoch_losses_mdn[1]
                                           
            
            if verbose:
                if i_epoch >=0:
                    viz.show_loss([epoch_loss_history[k_name][i_epoch] for k_name in c.loss_names], 
                                  logscale=False, its=min(len(test_loader), c.n_its_per_epoch) )
            
                    output_orig = output.cpu()
                    viz.show_hist(output_orig) # pass
    
            with torch.no_grad():
                samples = sample_z(1.)
                pass
            
            if (i_epoch > 10)*((i_epoch % 20)==0):
                # loss_array = np.array(epoch_loss_history)
                fig, ax = plot_loss_curve_2types(
                                        epoch_loss_history['Epoch'][:i_epoch+1],
                                        epoch_loss_history['Loss_train_mn'][:i_epoch+1],
                                        epoch_loss_history['Loss_test_mn'][:i_epoch+1],
                                        epoch_loss_history['Loss_train_mdn'][:i_epoch+1],
                                        epoch_loss_history['Loss_test_mdn'][:i_epoch+1],
                                        # loss_array[:,0], loss_array[:,1], loss_array[:,3], 
                                            c=c, 
                                          figname=c.filename+'_Loss_plot.pdf', # 이부분 hyperearch에선 바꿔줘야? 아님 나중에 폴더로 이동?
                                          yrange=c.loss_plot_yrange, xrange=c.loss_plot_xrange, 
                                          title=os.path.basename(c.config_file).replace('.py','') )
            
            model.model.zero_grad()
    
            if c.checkpoint_save:
                if (i_epoch % c.checkpoint_save_interval) == 0:
                    model.save(c.filename + '_checkpoint_%.4i' % (i_epoch * (1-c.checkpoint_save_overwrite)))
                    
            #####
            # Check convergence and divergence and change training_status
            #####
            if i_epoch > N_CONV_CHECK:
                flag_train_conv = train_tools.check_convergence(epoch_loss_history[name_check_train][:i_epoch+1], 
                                                   conv_cut=CONV_CUT, n_conv_check=N_CONV_CHECK)
                flag_test_conv = train_tools.check_convergence(epoch_loss_history[name_check_test][:i_epoch+1],
                                               conv_cut=CONV_CUT, n_conv_check=N_CONV_CHECK)
                if flag_train_conv * flag_test_conv:
                    if verbose:
                        print("Both train and test curves converged at: %d"%i_epoch)
                    training_status = 1 # 1=converged
                    
            if i_epoch > N_DIVG_CHECK:
                flag_train_divg = train_tools.check_divergence(epoch_loss_history[name_check_train][:i_epoch+1],
                                                   chunk_size=DIVG_CHUNK_SIZE, n_divg_check=N_DIVG_CHECK, divg_cri=DIVG_CRI)
                flag_test_divg = train_tools.check_divergence(epoch_loss_history[name_check_test][:i_epoch+1], 
                                                  chunk_size=DIVG_CHUNK_SIZE, n_divg_check=N_DIVG_CHECK, divg_cri=DIVG_CRI)
                if flag_test_divg==True or flag_train_divg==True:
                    if verbose:
                        print("Either train or test curve diverged at: %d"%i_epoch)
                    training_status = -1 # -1 = diverged
                    
            
        
        
        ### END LOOP    
        final_epoch = i_epoch + 1
        # remove unnecessary data in loss 
        epoch_loss_history = epoch_loss_history[:final_epoch]
        # update config with actural n_epochs used
        c.n_epochs = final_epoch
        train_tools.rewrite_config_element(c.config_file,  new_config_file=c.config_file, 
                                           param_to_change='n_epochs', value_to_change=c.n_epochs)
        
        if verbose:
            if c.live_visualization:
                viz.visualizer.viz.text('<h1 style="color:green">Done</h1>', win=running_box)
        model.save(c.filename)
        
        # loss_array = np.array(epoch_loss_history)
        fig, ax = plot_loss_curve_2types(
                                    epoch_loss_history['Epoch'][:i_epoch+1],
                                    epoch_loss_history['Loss_train_mn'][:i_epoch+1],
                                    epoch_loss_history['Loss_test_mn'][:i_epoch+1],
                                    epoch_loss_history['Loss_train_mdn'][:i_epoch+1],
                                    epoch_loss_history['Loss_test_mdn'][:i_epoch+1],
                                  c=c, 
                                  figname=c.filename+'_Loss_plot.pdf', #여기도 주의
                                  yrange=c.loss_plot_yrange, xrange=c.loss_plot_xrange, 
                                  title=os.path.basename(c.config_file).replace('.py',''))
        
        # header = 'Epoch\tL_train\tlr_train\tL_test\tlr_test'
        header = '\t'.join(epoch_loss_history.dtype.names)
        formats = []
        for name in epoch_loss_history.dtype.names:
            if epoch_loss_history[name].dtype == np.int32:
                formats.append('%d')
            else:
                formats.append('%.8f')
         
        np.savetxt(c.filename+'_loss_history.txt', epoch_loss_history, delimiter='\t', fmt='\t'.join(formats), header=header)

    except:
        model.save(c.filename + '_ABORT')
        loss_array = np.array(epoch_loss_history)
        if len(epoch_loss_history) > 10:
            fig, ax = plot_loss_curve_2types(
                                    epoch_loss_history['Epoch'][:i_epoch+1],
                                    epoch_loss_history['Loss_train_mn'][:i_epoch+1],
                                    epoch_loss_history['Loss_test_mn'][:i_epoch+1],
                                    epoch_loss_history['Loss_train_mdn'][:i_epoch+1],
                                    epoch_loss_history['Loss_test_mdn'][:i_epoch+1],
                                    c=c, 
                                      figname=c.filename+'_ABORT_Loss_plot.pdf',#여기도 주의
                                     title=os.path.basename(c.config_file).replace('.py',''))
        raise
    
    t_end = time()
    print('Time taken for training: {:.1f} hour ({:.1f} min)'.format( (t_end-t_start)/3600, (t_end-t_start)/60 ) )

    
    
    
    
"""
노이즈 넣어서 훈련하는 것 테스트. 최대한 train_network랑 비슷하게 만들어서 추후에는 하나로 합친다.

def train_noisy_network(c): # c is cINNConfig class variable

    # from cINN.viz import TrainVisualizer, plot_loss_curve
    
    t_start = time()
    
    config_str = ""
    config_str += "==="*30 + "\n"
    config_str += "Config options:\n\n"
    
    for param in c.parameter_list:
        config_str += "  {:25}\t{}\n".format(param, getattr(c, param))
    
    config_str += "==="*30 + "\n"
    print(config_str)
    
    # make model 
#    if c.model_code=='ModelAdamGLOW':
#        model = ModelAdamGLOW(c)
    model = eval(c.model_code)(c)
    
    # make savedir
    if not os.path.exists( os.path.dirname(c.filename) ):
        os.system('mkdir -p '+os.path.dirname(c.filename))
        
    # make viz
    viz = TrainVisualizer(c)
    if c.live_visualization:
        viz.visualizer.viz.text('<pre>' + config_str + '</pre>')
        running_box = viz.visualizer.viz.text('<h1 style="color:red">Running</h1>')
        
         
    if c.load_file:
        model.load(c.load_file, device=c.device)
        
    def sample_z(sigma, data_size):
    #    return sigma * torch.cuda.FloatTensor(c.batch_size, c.x_dim).normal_()
        return sigma * torch.FloatTensor(data_size, c.x_dim).normal_().to(c.device)
    
    try:
        epoch_loss_history = []
        data = DataLoader(c, update_rescale_parameters=True)
        
        if model.rescale_params is None:
            # pass rescale parameter information
            model.rescale_params = {'mu_x': data.mu_x, 'mu_y':data.mu_y, 'w_x':data.w_x, 'w_y':data.w_y}
            
        sig_array = torch.Tensor( [c.noise_fsigma[line] for line in c.y_names ] )
        
        for i_epoch in range(-c.pre_low_lr, c.n_epochs):
            
            test_loader, train_loader = data.get_loaders()
        
            data_iter = iter(train_loader)
            test_iter = iter(test_loader)
    
            loss_history = []
            test_loss_history = []
    
            if i_epoch < 0:
                for param_group in model.optim.param_groups:
                    param_group['lr'] = c.lr_init * 2e-2
                    
            # Loop for train_loader, Optimization
            for i_batch, data_tuple in tqdm.tqdm(enumerate(data_iter),
                                                  total=min(len(train_loader), c.n_its_per_epoch),
                                                  leave=False,
                                                  mininterval=1.,
                                                  disable=(not c.progress_bar),
                                                  ncols=83):
                
                # cpu torch.Tensor
                x, y = data_tuple
        
                ymin = 1 # for y'clipping
                
                # expand if needed
                if c.n_noise_MC > 1:
                    x = torch.repeat_interleave(x, c.n_noise_MC, dim=0) 
                    y = torch.repeat_interleave(y, c.n_noise_MC, dim=0)
                
                y = torch.clip( y*(1 + torch.randn(y.shape) * sig_array), min=ymin)
                # y = torch.clip(y, min=ymin) # y should > 0. use mininum of all line in this batch
                
                # Transform to rescaled: np array again
                x = torch.Tensor(data.params_to_x(x)).to(c.device)
                y = torch.Tensor(data.obs_to_y(y)).to(c.device)
                
       
                features = model.cond_net.features(y)
                
                if c.FrEIA_ver == 0.1:
                    output = model.model(x, features)
                    jac = model.model.log_jacobian(run_forward=False)
                elif c.FrEIA_ver == 0.2:
                    output, jac = model.model(x, features)
                else: # default = 0.1
                    output = model.model(x, features)
                    jac = model.model.log_jacobian(run_forward=False)
    
                zz = torch.sum(output**2, dim=1)   
                neg_log_likeli = 0.5 * zz - jac
    
                l = torch.mean(neg_log_likeli)
                l.backward(retain_graph=c.do_rev)
    
                if c.do_rev:
                    samples_noisy = sample_z(c.latent_noise, c.batch_size*c.n_noise_MC) + output.data
                    if c.FrEIA_ver == 0.1:
                        x_rec = model.model(samples_noisy, features, rev=True)
                    elif c.FrEIA_ver == 0.2:
                        x_rec, _ = model.model(samples_noisy, features, rev=True)
                    else:
                        x_rec = model.model(samples_noisy, features, rev=True)
                   
                    l_rev = torch.mean( (x-x_rec)**2 )
                    l_rev.backward()
                else:
                    l_rev = dummy_loss()
    
                model.optim_step()
                loss_history.append([l.item(), l_rev.item()])
    
                if i_batch+1 >= c.n_its_per_epoch:
                    # somehow the data loader workers don't shut down automatically
                    try:
                        data_iter._shutdown_workers()
                    except:
                        pass
    
                    break
        
            # Loop for test_loader, Get loss for test set
            for i_batch, test_tuple in tqdm.tqdm(enumerate(test_iter),
                                                  total=min(len(test_loader), c.n_its_per_epoch),
                                                  leave=False,
                                                  mininterval=1.,
                                                  disable=(not c.progress_bar),
                                                  ncols=83):
                # cpu torch.Tensor
                x, y = test_tuple
            
                ymin = 1 # for y'clipping
                
                # expand if needed
                if c.n_noise_MC > 1:
                    x = torch.repeat_interleave(x, c.n_noise_MC, dim=0) 
                    y = torch.repeat_interleave(y, c.n_noise_MC, dim=0)
                
                y = torch.clip( y*(1 + torch.randn(y.shape) * sig_array), min=ymin)
                # y = torch.clip(y, min=ymin) # y should > 0. use mininum of all line in this batch
                
                # Transform to rescaled: np array again
                x = torch.Tensor(data.params_to_x(x)).to(c.device)
                y = torch.Tensor(data.obs_to_y(y)).to(c.device)
                
    
                features = model.cond_net.features(y)
                with torch.no_grad():
                    if c.FrEIA_ver == 0.1:    
                        output = model.model(x, features)
                        jac = model.model.log_jacobian(run_forward=False)
                    elif c.FrEIA_ver == 0.2:
                        output, jac = model.model(x, features)
                    else:
                        output = model.model(x, features)
                        jac = model.model.log_jacobian(run_forward=False)
    
                zz = torch.sum(output**2, dim=1)
                neg_log_likeli = 0.5 * zz - jac
    
                tl = torch.mean(neg_log_likeli)
                # DO NOT BACKWARD FOR TEST!! USE NO_GRAD()
                # tl.backward(retain_graph=c.do_rev)
    
                if c.do_rev:
                    samples_noisy = sample_z(c.latent_noise) + output.data
                    with torch.no_grad():
                        if c.FrEIA_ver == 0.1:
                            x_rec = model.model(samples_noisy, features, rev=True)
                        elif c.FrEIA_ver == 0.2:
                            x_rec, _ = model.model(samples_noisy, features, rev=True)
                        else:
                            x_rec = model.model(samples_noisy, features, rev=True)
                    tl_rev = torch.mean( (x-x_rec)**2 )
                else:
                    tl_rev = dummy_loss()
                # tl_rev = dummy_loss()
    
                test_loss_history.append([tl.item(), tl_rev.item()])
    
                if i_batch+1 >= c.n_its_per_epoch:
                    # somehow the data loader workers don't shut down automatically
                    try:
                        test_iter._shutdown_workers()
                    except:
                        pass
    
                    break
                
            model.weight_scheduler.step()
            epoch_losses = np.mean(np.array(loss_history), axis=0)
            test_epoch_losses = np.mean(np.array(test_loss_history), axis=0)
            
            # save loss history per epoch w/ test loss 
            epoch_loss_history.append([i_epoch, epoch_losses[0], epoch_losses[1], 
                                       test_epoch_losses[0], test_epoch_losses[1]])

            
            if i_epoch > 2 - c.pre_low_lr:
                viz.show_loss([epoch_losses[0], test_epoch_losses[0], 
                               epoch_losses[1], test_epoch_losses[1]], 
                              logscale=False, its=min(len(test_loader), c.n_its_per_epoch) )
                output_orig = output.cpu()
                viz.show_hist(output_orig)
    
            with torch.no_grad():
                samples = sample_z(1., c.batch_size*c.n_noise_MC)
                pass
            
            if (i_epoch > 10)*((i_epoch % 20)==0):
                loss_array = np.array(epoch_loss_history)
                fig, ax = plot_loss_curve(loss_array[:,0], loss_array[:,1], loss_array[:,3], c=c, 
                                          figname=c.filename+'_Loss_plot.pdf', 
                                          yrange=c.loss_plot_yrange, xrange=c.loss_plot_xrange, 
                                          title=os.path.basename(c.config_file).replace('.py',''))
            
            model.model.zero_grad()
    
            if (i_epoch % c.checkpoint_save_interval) == 0:
                model.save(c.filename + '_checkpoint_%.4i' % (i_epoch * (1-c.checkpoint_save_overwrite)))
        
        
            
        if c.live_visualization:
            viz.visualizer.viz.text('<h1 style="color:green">Done</h1>', win=running_box)
        model.save(c.filename)
        
        loss_array = np.array(epoch_loss_history)
        fig, ax = plot_loss_curve(loss_array[:,0], loss_array[:,1], loss_array[:,3], c=c, 
                                  figname=c.filename+'_Loss_plot.pdf', yrange=c.loss_plot_yrange, xrange=c.loss_plot_xrange,
                                  title=os.path.basename(c.config_file).replace('.py',''))
        
        header = 'Epoch\tL_train\tlr_train\tL_test\tlr_test'
        np.savetxt(c.filename+'_loss_history.txt', np.array(epoch_loss_history), delimiter='\t', header=header)

    except:
        model.save(c.filename + '_ABORT')
        loss_array = np.array(epoch_loss_history)
        if len(loss_array) > 10:
            fig, ax = plot_loss_curve(loss_array[:,0], loss_array[:,1], loss_array[:,3], c=c,
                                      figname=c.filename+'_ABORT_Loss_plot.pdf',
                                     title=os.path.basename(c.config_file).replace('.py',''))
        raise
    
    t_end = time()
    print('Time taken for train: {:.1f} hour ({:.1f} min)'.format( (t_end-t_start)/3600, (t_end-t_start)/60 ) )

"""


def get_posterior(y_it, c, N=4096, return_llike=False, quiet=False):
    
    # import torch
    if c.network_model is None:
        c.load_network_model()
    model = c.network_model
    
    # model = eval(c.model_code)(c)
    # model.load(c.filename, device=c.device)
    
    if not quiet:
        print('Use %s'%(c.model_code))
        print('Use %s'%(c.filename))

    # y_it have to be an 1D/2D torch tensor/list/np array (c.y_dim_in or (# of y) x c.y_dim)
    y_it = torch.Tensor(y_it).reshape(-1, c.y_dim_in) # Make 2D torch Tensor even if you gave only 1D tensor (1, c.y_dim_in)
    outputs = []
    
    # if return_llike = True, it will calculate llike and return both outputs and llike
    # Or you can calculate llike separately by using get_loglikelihood function
    if return_llike:
        llikes = []
        
    for num, y in enumerate(y_it):

        features = model.cond_net.features(y.view(1,-1).to(c.device)).view(1, -1).expand(N, -1)
        z = torch.randn(N, c.x_dim).to(c.device)

        with torch.no_grad():
            x_samples, _ =  model.model(z, features, rev=True)
        
        outputs.append(x_samples.data.cpu().numpy())
        
        if return_llike:
            with torch.no_grad():
                _, jac =  model.model(x_samples, features, rev=False)
            
            zz = torch.sum(z**2., dim=1)
            log_likelihood = -(0.5*zz) + jac
            llikes.append( log_likelihood.to('cpu').detach().numpy() )
            

    # return a 3D ndarray (each array is N x c.x_dim) 
    # but this is x!!! you have to use astro.x_to_params -> this will automatically change list to np array
    if return_llike:
        return np.array(outputs), np.array(llikes)
    else:
        return np.array(outputs)
    
def get_posterior_group(y_it, c, N=4096, group=None, return_llike=False, quiet=False):
    
    # import torch
    if c.network_model is None:
        c.load_network_model()
    model = c.network_model
    
    if not quiet:
        print('Use %s'%(c.model_code))
        print('Use %s'%(c.filename))

    # y_it have to be an 1D/2D torch tensor/list/np array (c.y_dim_in or (# of y) x c.y_dim)
    y_it = torch.Tensor(y_it).reshape(-1, c.y_dim_in) # Make 2D torch Tensor even if you gave only 1D tensor (1, c.y_dim_in)
    
    # if return_llike = True, it will calculate llike and return both outputs and llike
    # Or you can calculate llike separately by using get_loglikelihood function
    if return_llike:
        llikes = []
        
    # g_max = int(1e8/(c.x_dim*2+c.y_dim_in)/N)
    g_max = int(4e6/N)
    if g_max < 1: g_max = 10
    if group is None:
        group = g_max
    elif group > g_max:
        group = g_max
        
    
    
    n_group = y_it.shape[0]//group
    if y_it.shape[0]%group > 0:
        n_group += 1
    
    for i_group in range(n_group):
        
        y = y_it[group*i_group: group*(i_group+1)]
        ny = y.shape[0]
        
        features = model.cond_net.features(y.to(c.device)).view(ny,1,-1).expand(-1, N,-1).reshape(ny*N,-1)
        z = torch.randn(features.shape[0], c.x_dim).to(c.device)
        
        with torch.no_grad():
            x_samples, _ =  model.model(z, features, rev=True)
            
        output = x_samples.data.cpu().numpy().reshape(ny, N, -1)
        if i_group==0:
            outputs = output
        else:
            outputs = np.vstack((outputs, output))
            
            
        if return_llike:
            with torch.no_grad():
                _, jac =  model.model(x_samples, features, rev=False)
            
            zz = torch.sum(z**2., dim=1)
            log_likelihood = -(0.5*zz) + jac
            llike =  log_likelihood.to('cpu').detach().numpy().reshape(ny, -1)
            
            if i_group==0:
                llikes = llike
            else:
                llikes = np.vstack((llikes, llike))

    # return a 3D ndarray (each array is N x c.x_dim)
    # but this is x!!! you have to use astro.x_to_params -> this will automatically change list to np array
    if return_llike:
        return np.array(outputs), np.array(llikes)
    else:
        return np.array(outputs)
    
    
# It is better to use get_posterior and calculate both posterior and llike at once - reduce error and computational time
# Error ~ 1e-4 (|llike from get_posterior - llike from get_log_likelihood|)
def get_log_likelihood(y_it, x_it, c, quiet=False):
        
    if c.network_model is None:
        c.load_network_model()
    model = c.network_model
    
    if not quiet:
        print('Use %s'%(c.model_code))
        print('Use %s'%(c.filename))
    
    y_it = torch.Tensor(y_it).to(c.device).reshape(-1, c.y_dim_in) # Make 2D torch Tensor even if you gave only 1D tensor (1, c.y_dim_in)
    # y_it (n, c.y_dim_in)
    # x_it (n, Nmodels, c.x_dim)
    llikes = []
    for num, y in enumerate(y_it):
 
        x = torch.Tensor(x_it[num]).to(c.device)
        features = model.cond_net.features(y.view(1,-1).to(c.device)).view(1, -1).expand(len(x), -1)
        with torch.no_grad():
            z_used, jac = model.model(x, c=features, rev=False)

        zz = torch.sum(z_used**2., dim=1)
        log_likelihood = -(0.5*zz) + jac
        llikes.append(log_likelihood.to('cpu').detach().numpy())
    
    return np.array(llikes)



