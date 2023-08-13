#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 11:45:52 2022

@author: daeun

test tools
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import torch
from scipy import stats
import pandas
from scipy.special import erfinv
from .train_tools import check_divergence, check_convergence

testfile_suffix = {
    'loss': '_loss_history.txt', # set in train
    'calib': '_calib.dat', 
    'RMSE' : '_RMSE.dat',
    'z_test': '_z_test.csv',
    'TvP': '_TvP.pkl',
    'MAP': '_MAP.dat' ,
    }

testfigure_suffix = {
    'loss': '_Loss_plot.pdf',
    'z_cov': '_z_cov_pdf.pdf',
    'z_corr': '_z_corr_pdf.pdf',
    'z_qq': '_z_qqplot.pdf',
    'calib': '_calib.pdf',
    'TvP':  '_TvP.pdf', 
    'TvP_MAP': '_TvP_MAP.pdf',
    }




def calculate_z(model, astro, smoothing=True):
    """
    Calculate latent variable for latent variable tests
    
    Parameters
    ----------
    model : cINN model
        cINN model
    astro : DataLoader
        cINN dataloader that contatins DB
    smoothing : bool, optional
        smoothing parameters or not. If  you trained your cINN after smoothing,
        you need to set smoothing True
        If smoothing sigma in astro is None, than smoothing will be automatically off

    Returns
    -------
    z_all : np array
        NxM (N: # of test models, M: dimension of parameters).

    """
    # if you use median/average of restuls from all testste, obs clipping is necessary!!!
    # if astro.train_noisy_obs != True: # deprecated
    
    veil_flux = False
    extinct_flux = False
    if astro.random_parameters is not None:
        if "veil_r" in astro.random_parameters.keys():
            veil_flux = True
        if "A_V" in astro.random_parameters.keys():
            extinct_flux = True
    
    if astro.prenoise_training == True:
        test, train = astro.get_splitted_set(rawval=True, smoothing=smoothing,
                                             smoothing_sigma=astro.smoothing_sigma,
                                             normalize_flux=astro.normalize_flux, 
                                             normalize_total_flux=astro.normalize_total_flux, 
                                             normalize_mean_flux=astro.normalize_mean_flux,
                                             veil_flux = veil_flux, extinct_flux = extinct_flux, 
                                             random_seed=0,
                                             )
        x_test, y_test = test[0], test[1]
        
        x_test = astro.params_to_x(x_test)
        
        
        ymin = np.min(y_test, axis=1)
        
        sig =  astro.create_uncertainty(tuple(y_test.shape))
        y_test = np.clip( y_test * (1+ (10**(sig))*np.random.randn(*y_test.shape) ), a_min=ymin.reshape(-1,1), a_max=None ) # all line independent
        y_test = astro.obs_to_y(y_test)
        
        xT = torch.Tensor(x_test).to(astro.device)
        yT = torch.hstack((torch.Tensor(y_test), torch.Tensor(astro.unc_to_sig(10**sig)))).to(astro.device)
        
    elif astro.use_flag == True:
        test, train = astro.get_splitted_set(rawval=False, smoothing=smoothing,
                                             smoothing_sigma=astro.smoothing_sigma,
                                             normalize_flux=astro.normalize_flux, 
                                             normalize_total_flux=astro.normalize_total_flux, 
                                             normalize_mean_flux=astro.normalize_mean_flux,
                                             veil_flux = veil_flux, extinct_flux = extinct_flux, 
                                             random_seed=0,
                                             )
        x_test, y_test = test[0], test[1]
        
        f_test = astro.create_random_flag(y_test.shape[0])
        for i_flag, flag_name in enumerate(astro.flag_names):
            roi_off = f_test[:, i_flag] == 0.0
            y_test[roi_off][:,astro.flag_index_dic[flag_name]] = 0.0
        
        xT = torch.Tensor(x_test).to(astro.device)
        yT = torch.hstack((torch.Tensor(y_test), torch.Tensor(astro.flag_to_rf(f_test)))).to(astro.device)
        
    
    else:
        test, train = astro.get_splitted_set(rawval=False, smoothing=smoothing,
                                             smoothing_sigma=astro.smoothing_sigma,
                                             normalize_flux=astro.normalize_flux, 
                                             normalize_total_flux=astro.normalize_total_flux, 
                                             normalize_mean_flux=astro.normalize_mean_flux,
                                             veil_flux = veil_flux, extinct_flux = extinct_flux, 
                                             random_seed=0,
                                             )
        x_test, y_test = test[0], test[1]
        
        yT = torch.Tensor(y_test).to(astro.device)
        xT = torch.Tensor(x_test).to(astro.device)
        


    features = model.cond_net.features(yT)

    with torch.no_grad():
        zT, _ = model.model(xT, features, rev=False)
#        if astro.FrEIA_ver == 0.1:
#            zT = model.model(xT, features, rev=False)
#        elif astro.FrEIA_ver == 0.2:
#            zT, _ = model.model(xT, features, rev=False)
#        else:
#            zT = model.model(xT, features, rev=False)

    z_all = zT.data.cpu().numpy()
    
    return z_all


def plot_z(z_all, figname=None, corrlabel=True, legend=True, yrange1=None, yrange2=None, 
           covariance = True, cmap=cm.get_cmap("gnuplot"), color_letter='r'):
    
    """
     Plot lateent variable covariance matrix and probability distribution
    
     Parameters
     ----------
     z_all : np array
         NxM (N: # of test models, M: dimension of parameters).
     figname : str, optional
         Path to the firgure file. The default is None.
     corrlabel : bool, optional
         Write coefficient values in the matrix. The default is True.
     legend : bool, optional
         plot legend. The default is True.
     yrange1 : list or tuple, optional
         y range of p(z). The default is None.
     yrange2 : TYPE, optional
         y range of p(z) residual. The default is None.
     covariance : bool, optional
         Use correlation coeffient or covariance. The default is True.
     cmap : colormap, optional
         colormap of z correlation matrix. The default is cm.get_cmap("gnuplot").
     color_letter : str, optional
         color of corrlabel. The default is 'r'.

     Returns
     -------
     figure, figures axis, residual axis.

     """
        
    if covariance:
        corr = np.cov(z_all, rowvar=False)
    else:
        corr = np.corrcoef(z_all, rowvar=False)
        

    fig, axis = plt.subplots(1,2, figsize=[8.5, 4.5])
    ax = axis[0]
    ims=ax.imshow(corr, cmap=cmap, vmin=np.min(corr), vmax=1)
    ax.minorticks_off()
    ax.set_xticks(np.arange(corr.shape[1]))
    ax.set_yticks(np.arange(corr.shape[0]))
    ax.set_xticklabels([str(i) for i in np.arange(corr.shape[1])+1], fontsize='large')
    ax.set_yticklabels([str(i) for i in np.arange(corr.shape[0])+1], fontsize='large')
    # ax.set_yticklabels(astro.parameter_names)

    ax.set_xlim([-0.5, corr.shape[1]-0.5])
    ax.set_ylim([-0.5, corr.shape[0]-0.5][::-1])
    ax.tick_params(axis='both', which='major', top='on',right='on',direction='in')
    ax.set(xlabel='$z_{j}$', ylabel='$z_{i}$')

    if corr.shape[0] > 10:
        corrlabel = False

    if corrlabel:
        for ix in range(corr.shape[0]):
            for iy in range(corr.shape[1]):
                ax.text(ix,iy,'{:.2f}'.format(corr[ix,iy]), ha='center', va='center', color=color_letter)
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("top",size="5%", pad="2%")
    cbar= fig.colorbar(ims, cax=cbar_ax,orientation='horizontal')
    cbar_ax.minorticks_off()
    cbar_ax.tick_params(which='major', axis='x', labeltop='on', labelbottom='off', bottom='off', top='on', direction='inout',
                       labelsize='medium')
    cbar_ax.xaxis.set_ticks_position("top")

    # p(z)
    ax = axis[1]
    hist_dic = {}
    bins = 100; hrange=(-7,7)

    stdnormal = lambda a: np.exp(-0.5*a*a)/np.sqrt(2*np.pi)
    xx = np.linspace(hrange[0], hrange[1],200)
    # yy = np.exp(-0.5*xx*xx)/np.sqrt(2*np.pi)
    yy = stdnormal(xx)
    ax.plot(xx,yy, '-', lw=5, color='r', label='N(0,1)', alpha=0.3)
    line_color={}
    for i in range(z_all.shape[1]+1):
        if i == 0:
            yhis, xhis = np.histogram(z_all.ravel(), bins=bins, range=hrange, density=True)
        else:
            yhis, xhis = np.histogram(z_all[:,i-1], bins=bins, range=hrange, density=True)
            line=ax.step( 0.5*(xhis[:-1]+xhis[1:]), yhis, where='mid', lw=1, label='$z_{%d}$'%i )
            line_color[i] = line[0].get_color()
        hist_dic[i] = {'yhis':yhis,'xhis':xhis}

    xhis = hist_dic[0]['xhis']; yhis=hist_dic[0]['yhis']
    line=ax.plot(0.5*(xhis[:-1]+xhis[1:]), yhis, '-', lw=2, label='all', color='k')
    line_color[0] = line[0].get_color()

    if legend:
        ax.legend(fontsize='medium')
    ax.set_xlim([-5,5])
    if yrange1 is not None:
        ax.set_ylim(yrange1)
    ax.minorticks_on()
    ax.tick_params(axis='both',which='both',top='on',right='on',direction='in' )
    ax.tick_params(axis='both',which='major', labelsize='large' )
    ax.set(ylabel='$p(z)$')
    #ax.grid()

    divider = make_axes_locatable(ax)
    res_ax = divider.append_axes("bottom",size="20%", pad="1%")
    for i in range(z_all.shape[1]+1):
        xhis = hist_dic[i]['xhis']
        xp = 0.5*(xhis[:-1]+xhis[1:])
        resi = hist_dic[i]['yhis'] - stdnormal(xp)
        if i==0:
            label='all'
            res_ax.step(xp, resi, where='mid',label=label, color=line_color[i], lw=1.2)
        else:
            label='str(i)'
            res_ax.step(xp, resi, where='mid',label=label, color=line_color[i], lw=0.8)

    res_ax.set_xlim(ax.get_xlim())
    if yrange2 is not None:
        res_ax.set_ylim(yrange2)
    res_ax.set_xlabel('$z$', fontsize='x-large')
    res_ax.minorticks_on()
    res_ax.tick_params(axis='both',which='both',top='on',right='on',direction='in' )
    res_ax.tick_params(axis='both',which='major', labelsize='medium' )
#     res_ax.tick_params(axis='x',which='minor',top=False,bottom=False)
    res_ax.grid(which='both', lw=0.5)

    fig.tight_layout()
    if figname:
        fig.savefig(figname,dpi=250)
        plt.close()
    return (fig, axis, res_ax)



def latent_normality_tests(z, filename=None):
    '''
    Computes various statistic tests (Kolmogorvo Smirnov, Shapiro-Wilk, Anderson_darling)
    to verify whether the M query latent variables follow the target normal distribution. Also
    computes the RMSE of the covariance matrix with respect to the target unity matrix.
    Saves all computed metrics to a txt file.

    Parameters
    ----------
    z : np.array
        N x M numpy array with predicted latent variables.
    ouput_dir : str, optional
        Path to csv file to save

    Returns
    -------
    result pandas data frame

    '''
    # Determine number of latent variables
    z_dim = z.shape[1]

    # Setup output array
    output = np.zeros(z_dim * 5 + 3) # Five statistics per variable + 3 for the covariance matrix

    # Compute RMSE for covariance matrix
    z_cov = np.cov(z, rowvar=False)
    ident = np.identity(n=z_dim)

    output[0] = np.sqrt(np.mean((z_cov - ident)**2))                                         # total RMSE
    output[1] = np.sqrt(np.mean((np.diagonal(z_cov) - np.diagonal(ident))**2))               # RMSE of diagonal elements
    output[2] = np.sqrt((np.sum(z_cov**2)-np.sum(np.diagonal(z_cov)**2))/(z_dim**2 - z_dim)) # RMSE of off-diagonal elements

    # Compute Kolmogorov Smirnov, Shapiro-Wilk and Anderson Darling test for every latent variable
    for n in range(z_dim):
        z_n = z[:,n]
        ks = stats.ks_1samp(z_n, stats.norm.cdf)
        sw = stats.shapiro(z_n)
        ad = stats.anderson(z_n)

        output[n*5 + 3] = ks.statistic      # KS statistic (Max distance between CDF, the smaller the better)
        output[n*5 + 3 + 1] = ks.pvalue     # KS p-value (0~1)
        output[n*5 + 3 + 2] = sw.statistic  # Shapiro-Wilk statistic (how close to normal distribution, should be close to 1)
        output[n*5 + 3 + 3] = sw.pvalue     # Shapiro-Wilk p-value
        output[n*5 + 3 + 4] = ad.statistic  # Anderson-Darling statistic (distance between distributions, the smaller the better)

    # Setup output header
    header = ["Cov_rmse_tot", "Cov_rmse_diag", "Cov_rmse_offdiag"]
    suffixes = ["ks_stat", "ks_p", "sw_stat", "sw_p", "ad_stat"]
    for i in range(z_dim):
        header += ["z_%i_%s" % (i, l) for l in suffixes]

    # Prepare data.frame and save to csv
    output_df = pandas.DataFrame(data=output.reshape(1, -1), columns=header)
    if filename is not None:
        output_df.to_csv(filename, index=False)
    
    return output_df



def qq_plot(z, figname=None, nCol=None, nRow=None, res=99):
    '''
    Plots quantile-quantile diagrams for the latent variables.

    Parameters
    ----------
    z : np.array
        N x M numpy array with the cINN output run in forward direction the test set,
        i.e. the latent variable distributions.
    figname : str, optional
        Path to figure file. if None, it returns figure
    nCol : int, optional
        Number of columns for the diagram.
    nRow : int, optinal
        Number of rows for the diagram.
    res : int, optional
        Resolution of the qq plot, i.e. number of quantiles computed between 0.01 and 0.99.
        The default is 99.
    
    Returns
    -------
    figure axis

    '''
    # Determine numbers of latent variables and determine number of rows
    z_dim = z.shape[1]
    
    if nCol is None and nRow is None:
        nRow = np.floor(np.sqrt(z_dim)).astype(int)
        
    if nCol is None:
        nCol = np.ceil(z_dim / nRow).astype(int)

    # Compute the empiric quantiles
    quants = np.linspace(0.01, 0.99, res)
    z_quants = np.quantile(z, q=quants, axis=0)

    # Compute the theoretical quantiles for a standard normal distribution
    # q = \sqrt(2) * erf^-1(2*p - 1)
    theo_quants = erfinv(2*quants-1) * np.sqrt(2)

    # Prepare plots
    f, axs = plt.subplots(nRow, nCol, figsize=(4*nCol, 4*nRow), tight_layout=True)
    
    ax1d = axs.ravel()
    for n in range(z_dim):
        ax = ax1d[n]

        ax.plot(theo_quants, theo_quants, c="red")
        ax.scatter(theo_quants, z_quants[:,n], c="black", s=5)
        # ax.tick_params(width=2, labelsize=15) 
        ax.set_xlabel("Theoretical quantiles", fontsize=14)
        ax.set_ylabel("Empirical quantiles", fontsize=14)
        # plt.setp(ax.spines.values(), linewidth=2) # ax테두리 두껍게
        ax.set_title('$z_%i$' % n, fontsize=19)
    
    if len(ax1d) > z_dim:
        for n in range(z_dim, len(axs)):
            ax1d[n].axis("off")
    # Save plot
    if figname is not None:
        f.savefig(figname, dpi=250)
        plt.close()
        
    return f, axs



def plot_calibration(calib_table, figname=None, return_figure=False):
    """
    Plot calibration and median uncertainty 

    Parameters
    ----------
    calib_table : astropy table
        calibration results.
    figname : str, optional
        figurename. The default is None.
    return_figure : bool, optional
        return figure and axis. The default is False.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    axis : TYPE
        DESCRIPTION.

    """
    
    n_param = int((len(calib_table.colnames)-1)/2)
    x_names = [a.replace('_clb_err','') for a in calib_table.colnames[1:1+n_param]]

    # nrow는 4의 배수로 끊김 ~4:1, ~8:2, ~12:3
    nrow = np.ceil(n_param/4).astype(int)
    ncol = np.ceil(n_param/nrow).astype(int)

    figsize = [3.1*ncol, 4*nrow]

    # if nrow==3:
    #     figsize[1] = 10
    # if ncol==3:
    #     figsize[0] = 8

    fig, axis = plt.subplots(nrow, ncol, figsize=figsize)
    axis_1d = axis.ravel()

    for i, param in enumerate(x_names):
        ax = axis_1d[i]

        xval = calib_table['confidence']
        yval = calib_table[param+'_clb_err']

        ax.plot(xval, yval)
        ax.axhline(y=0,ls='--', color='k', lw=0.5)
        # l1=ax.axhline(y=np.median(yval), label='Med: {:.3g}%'.format(np.median(yval)*100), ls='--', color='r', lw=1)
        # ax.legend(handles=[l1], fontsize='medium', loc='lower left')
        # l1=ax.axhline(y=np.median(yval), label='Median', ls='--', color='r', lw=1)

        ax.tick_params(labelsize='small', labelbottom=False)
        # if i==0:
        #     ax.legend(handles=[l1], fontsize='medium')
        ax.set_title(param, size='large')

        txt = [ ]
        txt.append( '$e^{\mathrm{med}}_{\mathrm{cal}}$ = '+'{:.3g}%'.format(np.median(abs(yval)*100)) )
        txt.append( '$e^{68\mathrm{conf}}_{\mathrm{cal}}$ = '+'{:.3g}%'.format(100*yval.data[67]) )
        txt.append(  '$\sigma^{68\mathrm{conf}}_{\mathrm{med}}$ = '+ '{:.3g}'.format(calib_table[param+'_unc_intrv'].data[67]) )
        ax.text(0.98, 0.97, '\n'.join(txt),
                        transform=ax.transAxes, ha='right', va='top', fontsize=8,
                         bbox=dict(boxstyle='square', facecolor='w', alpha=0.5, edgecolor='silver') )

        divider = make_axes_locatable(ax)
        ax1 = divider.append_axes("bottom", size="80%", pad=0.1, sharex=ax)
        yval = calib_table[param+'_unc_intrv']
        ax1.plot(xval, yval)

        ax1.set_xlabel('Confidence', size='medium')
        ax1.tick_params(axis='both', which='both', labelsize='small')

        if i%ncol==0:   
            ax1.set_ylabel('Median Uncertainty', size='medium')
            ax.set_ylabel('Calibration error', size='medium')



    i+=1
    for j in range(i, len(axis_1d)):
        axis_1d[j].axis("off")

    fig.tight_layout()
    
    if figname is not None:
        fig.savefig(figname, dpi=250)
        plt.close()
        
    if return_figure:
        return fig, axis

def check_training_status(c):
    
    name_check_train = 'Loss_train_mdn'
    name_check_test = 'Loss_test_mdn'
    try:
        epoch_loss_history = np.genfromtxt(c.filename+testfile_suffix['loss'], names=True)
        flag_train_conv = check_convergence(epoch_loss_history[name_check_train])
        flag_test_conv = check_convergence(epoch_loss_history[name_check_test])
        flag_train_divg = check_divergence(epoch_loss_history[name_check_train])
        flag_test_divg = check_divergence(epoch_loss_history[name_check_test])
        
        if flag_test_divg==True or flag_train_divg==True:
            training_status = -1 # Diverged
        elif flag_train_conv * flag_test_conv:
            training_status = 1 # Converged
        else:
            training_status = 0
               
        return training_status
               
    except:
        return None
    


def combine_evaluations(c, filename=None, info_dic=None, return_output=False, sep=','):
    """
    Combine evaluation result of one network:
        loss, calibration, latent tests

    Parameters
    ----------
    c : class
        config.
    filename : str, optional
        filepath. The default is None.
    info_dic : dict, optional
        name of each evaluation file. keep it None unless you set specific paths. The default is None.
    return_output : bool, optional
        return output dataframe. The default is False.
    sep : str, optional
        sep for df.to_csv(). The default is ','.

    Returns
    -------
    output_df : pandas df


    """

    if info_dic is None:
        info_dic = {}
        info_dic['RMSE'] = c.filename + testfile_suffix['RMSE']
        info_dic['calib'] = c.filename + testfile_suffix['calib']
        info_dic['loss'] = c.filename + testfile_suffix['loss'] # loss file
        info_dic['z_test'] = c.filename + testfile_suffix['z_test']
    
    # Loss information
    header = ['Loss_train_mn', 'Loss_test_mn', 'Loss_train_mdn', 'Loss_test_mdn']
    header += ['L_rev_train_mn', 'L_rev_test_mn', 'L_rev_train_mdn', 'L_rev_test_mdn']
    header += ['Training_status']
    
    
    name_check_train = 'Loss_train_mdn'
    name_check_test = 'Loss_test_mdn'
    try:
        epoch_loss_history = np.genfromtxt(info_dic['loss'], names=True)
        flag_train_conv = check_convergence(epoch_loss_history[name_check_train])
        flag_test_conv = check_convergence(epoch_loss_history[name_check_test])
        flag_train_divg = check_divergence(epoch_loss_history[name_check_train])
        flag_test_divg = check_divergence(epoch_loss_history[name_check_test])
        
        if flag_test_divg==True or flag_train_divg==True:
            training_status = -1 # Diverged
        elif flag_train_conv * flag_test_conv:
            training_status = 1 # Converged
        else:
            training_status = 0
               
        loss_data = pandas.read_csv(info_dic['loss'], sep='\t').values[-1, 1:]
        loss_data = np.append(loss_data, [training_status])
               
    except:
        loss_data = np.zeros(len(header)) + np.nan
        
    # calibration information and RMSE information
    for param in c.x_names:
        header += [param + a for a in ["_med_abs_e_cal", "_u68", "_rmse_map", "_rmse_all"]]
    header += ["mean_med_abs_e_cal", "mean_u68", "mean_rmse_map", "mean_rmse_all"]        

    try:
        calib_df = pandas.read_csv(info_dic['calib'], sep='\t')
        rmse_df = pandas.read_csv(info_dic['RMSE'], sep='\t')
    
        med_abs_e_data = np.nanmedian(np.abs(calib_df.values[:,1:len(c.x_names)+1]), axis=0) # med_abs_e_cal
        u68_data = calib_df.values[67,-len(c.x_names):]
        rmse_map_data = rmse_df.values[0,1:].astype(float)
        rmse_all_data = rmse_df.values[2,1:].astype(float)

        data = np.stack( (med_abs_e_data, u68_data, rmse_map_data, rmse_all_data), axis=-1)
        eval_data = np.concatenate( (data, np.nanmean(data, axis=0).reshape(1,-1))).ravel()
    except:
        eval_data = np.zeros( (len(c.x_names)+1)*4) + np.nan
        
        
    # latent variable tests
    header += ["Cov_rmse_tot", "Cov_rmse_diag", "Cov_rmse_offdiag"]
    # Z normaility tests
    for i in range(c.x_dim):
        header += ["z_%i" % i + a for a in ["_ks_stat", "_ks_p", "_sw_stat", "_sw_p", "_ad_stat"]]
    
    try:
        z_data = (pandas.read_csv(info_dic['z_test'])).values.reshape(-1)
    except:
        z_data = np.zeros( 3+c.x_dim*5 ) + np.nan
        
    output_df = pandas.DataFrame( np.concatenate((loss_data, eval_data, z_data)).reshape(1,-1), columns=header)
    
    if filename is not None:
        output_df.to_csv(filename, index=False, na_rep='NULL', sep=sep)

    if return_output:
        return output_df
    
    
def combine_multiple_evaluations(config_list, filename=None, sep=',', return_output=False):
    """
    Combine evaluations of multiple networks

    Parameters
    ----------
    config_list : list
        List of config class.
    filename : str, optional
        Final filepath. The default is None.
    return_output : bool, optional
        return output dataframe. The default is False.
    sep : str, optional
        sep for df.to_csv(). The default is ','.

    Returns
    -------
    output_df : pandas df

    """
    for i, c in enumerate(config_list):
        output_i = combine_evaluations(c, filename=None, return_output=True) # df 
        if i==0:
            output = output_i
        else:
            output = pandas.concat( [output, output_i],  axis=0, ignore_index=True)
            
            
    if filename is not None:
        output.to_csv(filename, index=False, sep=sep)   
    
    if return_output:
        return output
    
    
    
def check_train_status(c):
    files_to_check = [c.filename, c.filename + testfile_suffix['loss'], c.filename + testfigure_suffix['loss'] ]
    train_status = True
    for file in files_to_check:
        train_status *= os.path.exists(file)
    return train_status

def check_eval_status(c):
    files_to_check = [c.filename + testfile_suffix['z_test'], c.filename + testfigure_suffix['z_cov'],
                 c.filename + testfigure_suffix['z_corr'], c.filename + testfigure_suffix['z_qq'] ]

    files_to_check += [c.filename + testfile_suffix['calib'], c.filename + testfile_suffix['RMSE'],
                      c.filename + testfile_suffix['TvP'], c.filename + testfile_suffix['MAP'],   
                      c.filename + testfigure_suffix['calib'], c.filename + testfigure_suffix['TvP'],
                     c.filename + testfigure_suffix['TvP_MAP'], ]

    eval_status = True
    for file in files_to_check:
        eval_status *= os.path.exists(file)
    return eval_status   
    
    
    
    
def clean_evalfiles(c):
    files_to_clean = [c.filename + testfile_suffix['z_test'], c.filename + testfigure_suffix['z_cov'],
                 c.filename + testfigure_suffix['z_corr'], c.filename + testfigure_suffix['z_qq'] ]

    files_to_clean += [c.filename + testfile_suffix['calib'], c.filename + testfile_suffix['RMSE'],
                      c.filename + testfile_suffix['TvP'], c.filename + testfile_suffix['MAP'],   
                      c.filename + testfigure_suffix['calib'], c.filename + testfigure_suffix['TvP'],
                     c.filename + testfigure_suffix['TvP_MAP'], ]

    for file in files_to_clean:
        os.system('rm -rf %s'%file)
    
    
    
    
    
    
    
