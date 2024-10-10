# 2022. 1. 10. Ver007 FrEIA version 0.1, 0.2

#!/usr/bin/env python
# import sys
import os
import torch
import torch.nn
import torch.optim
from torch.nn.functional import avg_pool2d, interpolate
from torch.autograd import Variable
import numpy as np
import tqdm
# import matplotlib
# import matplotlib.pyplot
# matplotlib.use('Agg')

from time import time

from .models import * # ModelAdamGlow
from .data_loader import DataLoader
from .viz import *
from .tools import train_tools

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

"""
Modified train_network (2022.08.30)
- verbose, etc sentences
"""
def train_network(c, data=None, verbose=True, max_epoch=1000): # c is cINNConfig class variable

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
        # epoch_loss_history = []
        
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
            model.rescale_params = {'mu_x': data.mu_x, 'mu_y':data.mu_y, 'w_x':data.w_x, 'w_y':data.w_y}
            
        
        ########
        # training status: -1, 0 or 1
        # 0: training in progress (not converged, not diverged)
        # 1: training converged (both train set and test set converged)
        # -1: training diverged (either train set or test set diverged)
        training_status = 0
        i_epoch = -c.pre_low_lr - 1
        
        name_check_train = 'Loss_train_mdn'
        name_check_test = 'Loss_test_mdn'
        
        while (training_status == 0 and i_epoch < max_epoch-1):
            i_epoch += 1
        
            test_loader, train_loader = data.get_loaders( param_seed = i_epoch) # param_seed will randomize parameter if random_parameters is set
        
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
    
                x, y = data_tuple
                x, y = x.to(c.device), y.to(c.device)
                
                features = model.cond_net.features(y)
                
#                if c.FrEIA_ver == 0.1:
#                    output = model.model(x, features)
#                    jac = model.model.log_jacobian(run_forward=False)
#                elif c.FrEIA_ver == 0.2:
#                    output, jac = model.model(x, features)
#                else: # default = 0.1
#                    output = model.model(x, features)
#                    jac = model.model.log_jacobian(run_forward=False)
                # from 2023.08.10. FrEIA_ver deprecated: used FrEIA>0.2 saved in cINN_set
                output, jac = model.model(x, features)
                
                zz = torch.sum(output**2, dim=1)
    
                neg_log_likeli = 0.5 * zz - jac
    
                l = torch.mean(neg_log_likeli)
                l.backward(retain_graph=c.do_rev)
    
                if c.do_rev:
                    samples_noisy = sample_z(c.latent_noise) + output.data
                    # original code raised AssertionError (no condition input)
                    # x_rec = model.model(samples_noisy, rev=True)
#                    if c.FrEIA_ver == 0.1:
#                        x_rec = model.model(samples_noisy, features, rev=True)
#                    elif c.FrEIA_ver == 0.2:
#                        x_rec, _ = model.model(samples_noisy, features, rev=True)
#                    else:
#                        x_rec = model.model(samples_noisy, features, rev=True)
                    # from 2023.08.10. FrEIA_ver deprecated: used FrEIA>0.2 saved in cINN_set
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
                x, y = x.to(c.device), y.to(c.device)
    
                features = model.cond_net.features(y)
                with torch.no_grad():
#                    if c.FrEIA_ver == 0.1:
#                        output = model.model(x, features)
#                        jac = model.model.log_jacobian(run_forward=False)
#                    elif c.FrEIA_ver == 0.2:
#                        output, jac = model.model(x, features)
#                    else:
#                        output = model.model(x, features)
#                        jac = model.model.log_jacobian(run_forward=False)
                    output, jac = model.model(x, features)
                        
                zz = torch.sum(output**2, dim=1)
                neg_log_likeli = 0.5 * zz - jac
    
                tl = torch.mean(neg_log_likeli)
                # DO NOT BACKWARD FOR TEST!! USE NO_GRAD()
                # tl.backward(retain_graph=c.do_rev)
    
                if c.do_rev:
                    samples_noisy = sample_z(c.latent_noise) + output.data
                    with torch.no_grad():
#                        if c.FrEIA_ver == 0.1:
#                            x_rec = model.model(samples_noisy, features, rev=True)
#                        elif c.FrEIA_ver == 0.2:
#                            x_rec, _ = model.model(samples_noisy, features, rev=True)
#                        else:
#                            x_rec = model.model(samples_noisy, features, rev=True)
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
            # print(test_epoch_losses_mn)
            
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
                if i_epoch > 2 - c.pre_low_lr:
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
        i_epoch = -c.pre_low_lr - 1
        
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
                if i_epoch > 2 - c.pre_low_lr:
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
        i_epoch = -c.pre_low_lr - 1
        
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
                if i_epoch > 2 - c.pre_low_lr:
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
        i_epoch = -c.pre_low_lr - 1
        
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
                if i_epoch > 2 - c.pre_low_lr:
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
        i_epoch = -c.pre_low_lr - 1
        
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
                if i_epoch > 2 - c.pre_low_lr:
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



