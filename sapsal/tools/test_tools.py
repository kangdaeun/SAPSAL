#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 11:45:52 2022

@author: daeun

test tools
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.cm as cm
# import matplotlib.colormaps as cm
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as PathEffects
import os
import torch
from scipy import stats
import pandas
from scipy.special import erfinv
import copy
from .train_tools import check_divergence, check_convergence
import warnings
# warnings.filterwarnings("ignore", category=UserWarning)


testfile_suffix = {
    'loss': '_loss_history.txt', # set in train
    'calib': '_calib.dat', 
    'RMSE' : '_RMSE.dat',
    'z_test': '_z_test.csv',
    'TvP': '_TvP.pkl',
    'MAP': '_MAP.dat' ,
    'u68': '_u68.dat'
    }

testfigure_suffix = {
    'loss': '_Loss_plot.pdf',
    'z_cov': '_z_cov_pdf.pdf',
    'z_corr': '_z_corr_pdf.pdf',
    'z_qq': '_z_qqplot.pdf',
    'calib': '_calib.pdf',
    'TvP':  '_TvP.pdf', 
    'TvP_MAP': '_TvP_MAP.pdf',
    # 'Ddist': '_Ddist.pdf'
    }


def load_data_for_situation(astro, smoothing=False, random_seed=0):
    """
    Return rescaled x_test and y_test. y_test includes all information to feed into cond_net (e.g. sig, flag)
    (numpy array)
    
    """
    
    # available options:
    # z_test, D_cal, tSNE, pred
    
    # if you use median/average of restuls from all testste, obs clipping is necessary!!!
    # if astro.train_noisy_obs != True: # deprecated
    
    veil_flux = False
    extinct_flux = False
    if astro.random_parameters is not None:
        if "veil_r" in astro.random_parameters.keys():
            veil_flux = True
        if "A_V" in astro.random_parameters.keys():
            extinct_flux = True
            
    # smoothing is different for situation:
    # for z_test: do smoothing
    # for prediction: do not smooth
    # for D distribution: not needed. does not matter. smoothing=False
    
  
    # Generally, situation depends most on network setup
    test, train = astro.get_splitted_set(rawval=False, smoothing=smoothing,
                                         smoothing_sigma=astro.smoothing_sigma,
                                         normalize_flux=astro.normalize_flux, 
                                         normalize_total_flux=astro.normalize_total_flux, 
                                         normalize_mean_flux=astro.normalize_mean_flux,
                                         veil_flux = veil_flux, extinct_flux = extinct_flux, 
                                         random_seed=random_seed,
                                         )
    x_test, y_test = test[0], test[1]
    
    if astro.prenoise_training:
        # change y to obs and make sigma
        y_test = astro.y_to_obs(y_test)
        ymin = np.min(y_test, axis=1)
        sig =  astro.create_uncertainty(tuple(y_test.shape))
        y_test = np.clip( y_test * (1+ (10**(sig))*np.random.randn(*y_test.shape) ), a_min=ymin.reshape(-1,1), a_max=None ) # all line independent
        
        y_test = np.hstack( [astro.obs_to_y(y_test), astro.unc_to_sig(10**sig)] )
        
    elif astro.use_flag:
        # make flag and change y
        f_test = astro.create_random_flag(y_test.shape[0])
        for i_flag, flag_name in enumerate(astro.flag_names):
            roi_off = f_test[:, i_flag] == 0.0
            y_test[roi_off][:,astro.flag_index_dic[flag_name]] = 0.0
            
    elif astro.wavelength_coupling:
        wl_test = astro.create_coupling_wavelength(y_test.shape[0]) 
        # Transform to rescaled: wl - lambda & torch Tensor
        lam = astro.wl_to_lambda(wl_test)
        # permute (flux and wl together)
        perm = np.random.permutation(y_test.shape[1])
        y_test = np.hstack( [ y_test[:,perm], lam[:,perm] ] )
        
    else:
        pass
    
    return (x_test, y_test) 
    # xy in rescaled. y_test include all to feed into cond_net
       
        
 


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
    # in z_test. smoothing should be same as c.smoothing 
    x_test, y_test = load_data_for_situation(astro, smoothing=smoothing)
    
    # change to torch and device
    xT = torch.Tensor(x_test).to(astro.device)
    yT = torch.Tensor(y_test).to(astro.device)
    
    features = model.cond_net.features(yT)

    with torch.no_grad():
        zT, _ = model.model(xT, features, rev=False)
#        if astro.FrEIA_ver == 0.1:
#            zT = model.model(xT, features, rev=False)
    z_all = zT.data.cpu().numpy()
    
    return z_all


def plot_z(z_all, figname=None, corrlabel=True, legend=True, yrange1=None, yrange2=None, 
           covariance = True, cmap=plt.get_cmap("gnuplot"), color_letter='r', return_figure=False,
           title=None, titlesize='large'):
    
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
         colormap of z correlation matrix. The default is plt.get_cmap("gnuplot").
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
            line=ax.step( 0.5*(xhis[:-1]+xhis[1:]), yhis, where='mid', lw=1, label='$z_{%d}$'%(i-1) )
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
    res_ax.axhline(y=0, zorder=0, color='grey', lw=0.5, ls='--')
    res_ax.set_xlim(ax.get_xlim())
    if yrange2 is not None:
        res_ax.set_ylim(yrange2)
    res_ax.set_xlabel('$z$', fontsize='x-large')
    res_ax.minorticks_on()
    res_ax.tick_params(axis='both',which='both',top='on',right='on',direction='in' )
    res_ax.tick_params(axis='both',which='major', labelsize='medium' )
#     res_ax.tick_params(axis='x',which='minor',top=False,bottom=False)
    res_ax.grid(which='both', lw=0.5)
    
    if title is not None:
        fig.suptitle(title, fontsize=titlesize)

    fig.tight_layout()
    if figname:
        fig.savefig(figname,dpi=250)
        plt.close()
    if return_figure:
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



def qq_plot(z, figname=None, nCol=None, nRow=None, res=99,
            title=None, titlesize='large'):
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
        for n in range(z_dim, len(ax1d)):
            ax1d[n].axis("off")
            
    if title is not None:
        f.suptitle(title, fontsize=titlesize)
        
    # Save plot
    if figname is not None:
        f.savefig(figname, dpi=250)
        plt.close()
        
    return f, axs



def plot_calibration(calib_table, figname=None, return_figure=False, title=None, titlesize='large'):
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

    figsize = [3.1*ncol, 4.1*nrow]

    # if nrow==3:
    #     figsize[1] = 10
    # if ncol==3:
    #     figsize[0] = 8

    fig, axis = plt.subplots(nrow, ncol, figsize=figsize, tight_layout=1)
    axis_1d = axis.ravel()

    for i, param in enumerate(x_names):
        ax = axis_1d[i]

        xval = calib_table['confidence']
        yval = calib_table[param+'_clb_err']

        ax.plot(xval, yval)
        ax.axhline(y=0,ls='--', color='k', lw=0.5)

        ax.tick_params(labelsize='small', labelbottom=False)
        ax.set_title(param, size='large')

        txt = [ ]
        txt.append( r'$e^{\mathrm{med}}_{\mathrm{cal}}$ = '+'{:.3g}%'.format(np.median(abs(yval)*100)) )
        txt.append( r'$e^{68\mathrm{conf}}_{\mathrm{cal}}$ = '+'{:.3g}%'.format(100*yval.data[67]) )
        txt.append(  r'$\sigma^{68\mathrm{conf}}_{\mathrm{med}}$ = '+ '{:.3g}'.format(calib_table[param+'_unc_intrv'].data[67]) )
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

    if title is not None:
        fig.suptitle(title, fontsize=titlesize)
    
    if figname is not None:
        fig.savefig(figname, dpi=250)
        plt.close()
        
    if return_figure:
        return fig, axis
    
   
# plot_TvP(hist_dic, c,  rmse_table, N_test=len(obs_test), discretized_parameters=discretized_parameters, plotting_map=False, figname=None, return_figure=False)
def plot_TvP(hist_dic, c, rmse_table, N_test, plotting_map=False, discretized_parameters=None,
             figname=None, return_figure=False, title=None, titlesize='large'):

    """
    Calculate confusion matrix for discretized parameters (All)
    """
    conf_matrix = {}

    for param in discretized_parameters:
        if param in c.x_names:
            H, xedges, yedges = hist_dic[param]['H'].copy(), hist_dic[param]['xedges'], hist_dic[param]['yedges']
            H = H.transpose()
            conf = H.copy()*0.0

            for i in range(H.shape[1]):
                conf[:,i] = H[:,i]/np.sum(H[:,i])*100

            conf_matrix[param] = conf

    if plotting_map:
        H_min = 5 # 보통
        if N_test < 1e4:
            H_min = 1
    else: # all posterior 
        H_min = 10 # 보통
        if N_test < 1e4:
            H_min = 5
        
    
    # set figure size and grid
    # nrow는 4의 배수로 끊김 ~4:1, ~8:2, ~12:3
    nrow = np.ceil(len(c.x_names)/4).astype(int)
    ncol = np.ceil(len(c.x_names)/nrow).astype(int)
    # figsize = [3.1*ncol, 4*nrow]
    figsize = [3.3*ncol, 3.8*nrow]
    fig, axis = plt.subplots(nrow, ncol, figsize=figsize, tight_layout=1)
    axis = axis.ravel()

    for i_param, param in enumerate(c.x_names):
        ax = axis[i_param]

        H, xedges, yedges = hist_dic[param]['H'].copy(), hist_dic[param]['xedges'], hist_dic[param]['yedges']

        roi = H < H_min
        Hsum = np.nansum(H)
        H[roi] = np.nan
        H = H/Hsum

        # cmap = copy.copy(cm.gnuplot2_r)
        cmap = copy.copy(cm.Oranges)
        # cmap = copy.copy(cm.gnuplot)
        # cmap.set_bad("grey", 0.2)

        ims=ax.imshow( H.transpose(), origin='lower', extent = (xedges[0], xedges[-1], yedges[0], yedges[-1]), 
                      aspect='auto',
                 cmap=cmap)#, norm=clr.LogNorm())
        
        axins = ax.inset_axes( [0.07, 0.89, 0.5, 0.03 ])
        cbar=fig.colorbar(ims, cax=axins, orientation="horizontal")
        cbar.minorticks_off()
        axins.tick_params(axis='both',which='both',labelsize='medium', direction='out', length=2)
        axins.set_title('Fraction', size='medium')
        # axins.xaxis.set_major_locator(ticker.LogLocator(numticks=3))
        axins.xaxis.set_major_locator(ticker.MaxNLocator(3))

        if param in conf_matrix.keys():
            conf = conf_matrix[param]

            xp = 0.5*(xedges[1:]+xedges[:-1])
            yp = 0.5*(yedges[1:]+yedges[:-1])

            for ix, x in enumerate(xp):
                for iy, y in enumerate(yp):
                    if np.isfinite(H[ix,iy]):
                        string = '%.2g'%(conf[iy,ix])
                        if (conf[iy,ix] < 1): string = '%.1f'%(conf[iy,ix])
                        if conf[iy,ix] >= 99.5: string = '100'
                        ax.text(x, y, string,  color='k', fontsize=8.5, ha='center', va='center', path_effects=[PathEffects.withStroke(linewidth=2, foreground='w')] )
            # change ticks
            ax.set_xticks(xp)
            ax.set_yticks(yp)

        # 1:1 
        xr = ax.get_xlim()
        yr = ax.get_ylim()
        mini = np.min(xr+yr); maxi = np.max(xr+yr)
        xx=np.linspace(mini, maxi, 500)
        ax.plot(xx,xx,'-',color='navy', lw=0.7)
        ax.set_xlim(xr); ax.set_ylim(yr)
        
        exp = c.import_expander()
        # special tick in the case of SpTind
        if param == 'SpTind':
            # for M0, K0, G0, etc
            major_ticklabels = [f"{a}0" for a in ['O', 'B', 'A', 'F', 'G', 'K', 'M']]
            major_ticks = [ exp.convert_spt_to_num(spt) for spt in major_ticklabels ]
            minor_ticks = np.arange(major_ticks[0], major_ticks[-1]+9.5, 1)
            
            ax.xaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
            ax.yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
            ax.set_xticks(major_ticks, labels=major_ticklabels)
            ax.set_yticks(major_ticks, labels=major_ticklabels)
        
            ax.set_xlim(xr); ax.set_ylim(yr)
        
        if plotting_map:
            txt = 'RMSE = %.4g\n'%(rmse_table[param][rmse_table['type']=='RMSE_MAP_PARAM'][0])+r'N$_{\mathrm{test}}$ = %d'%(N_test)
        else:
            txt = 'RMSE = %.4g\n'%(rmse_table[param][rmse_table['type']=='RMSE_ALL_PARAM'][0])+r'N$_{\mathrm{test}}$ = %d'%(N_test)
        ax.text(0.95, 0.05, txt,  
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
                 bbox=dict(boxstyle='square', facecolor='w', alpha=1, edgecolor='silver') )

        if plotting_map:
            ax.set(title=param, xlabel=r'$X^{\mathrm{True}}$', ylabel=r"$X^{\mathrm{MAP}}$")
        else:
            ax.set(title=param, xlabel=r'$X^{\mathrm{True}}$', ylabel=r"$X^{\mathrm{Post}}$")

    for j in range(i_param+1, len(axis)):
        axis[j].axis("off")
        
    if title is not None:
        fig.suptitle(title, fontsize=titlesize)
        
    if figname is not None:
        fig.savefig(figname, dpi=250)
        plt.close()
        
    if return_figure:
        return fig, axis
        
    
def calculate_D(model, astro, smoothing=False, 
                y_test=None, y_real=None, 
                ):
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
    D_test, D_real : np array
        output of discriminator

    """
    if np.logical_or(y_test is None, y_real is None):
    
        x_test, y_test = load_data_for_situation(astro, smoothing=smoothing)
        # rescaled parameters. y include all thing
        y_test = torch.Tensor(y_test).to(astro.device)
        
        # Load real data
        if astro.prenoise_training:
            y_real, s_real =  astro.get_real_data(rawval=False, normalize_flux=astro.normalize_flux, 
                                                         normalize_total_flux=astro.normalize_total_flux, 
                                                         normalize_mean_flux=astro.normalize_mean_flux,
                                                  )
            y_real = np.hstack( [y_real, s_real] )
        
        else:
            y_real =  astro.get_real_data(rawval=False, normalize_flux=astro.normalize_flux, 
                                                         normalize_total_flux=astro.normalize_total_flux, 
                                                         normalize_mean_flux=astro.normalize_mean_flux,
                                                         )
            
        y_real = torch.Tensor(y_real).to(astro.device)
    else:
        y_test = torch.Tensor(y_test).to(astro.device)
        y_real = torch.Tensor(y_real).to(astro.device)
    
    with torch.no_grad():
        features_test = model.cond_net.features(y_test)
        D_test = model.da_disc(features_test)
        
        features_real = model.cond_net.features(y_real)
        D_real = model.da_disc(features_real)
    
        D_real = torch.sigmoid(D_real).data.cpu().numpy().ravel()
        D_test = torch.sigmoid(D_test).data.cpu().numpy().ravel()
        
    return (D_test, D_real)


def plot_D_distribution(D_test, D_real, figname=None, return_figure=False,
                        title=None, titlesize='large', 
                        test_color = 'C1', real_color='C2', xylabelsize='large'
                            ):
    
    vv = np.hstack([D_test,D_real])
    hist_range = (np.nanmin(vv), np.nanmax(vv))
    fig, ax = plt.subplots(1,1, figsize=[5, 4.], tight_layout=1)
    kwarg = {'density':True, 'bins':100, 'alpha':0.5, 'range':hist_range}
    for val, label in zip([D_test, D_real], ['Dtest','Dreal']):
        avg = np.mean(val); std=np.std(val)
        txt = label + r': %.1f$\pm$%.1f'%(avg, std)
        _ = ax.hist(val, label=txt, **kwarg)
        
    ax.legend()
    ax.set_ylabel('Probability density', size=xylabelsize)
    ax.set_xlabel('Discriminator(condition)', size=xylabelsize)
    if title is not None:
        ax.set_title(title, size=titlesize)
    
    if figname is not None:
        fig.savefig(figname, dpi=250)
        plt.close()
        
    if return_figure:
        return fig, ax
    
    
def calculate_tSNE(model, astro, smoothing=False, do_pca = False, n_components_pca = 50,
                   n_components=2, perplexity=30, random_state=42, 
                   cval_name = None,
                   y_test=None, y_real=None, x_test=None,
                   ):
    
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    if y_test is None or y_real is None or x_test is None:
        x_test, y_test = load_data_for_situation(astro, smoothing=smoothing)
        # rescaled parameters. y include all thing
        y_test = torch.Tensor(y_test).to(astro.device)
        
        # Load real data
        if astro.prenoise_training:
            y_real, s_real =  astro.get_real_data(rawval=False, normalize_flux=astro.normalize_flux, 
                                                         normalize_total_flux=astro.normalize_total_flux, 
                                                         normalize_mean_flux=astro.normalize_mean_flux,
                                                  )
            y_real = np.hstack( [y_real, s_real] )
        
        else:
            y_real =  astro.get_real_data(rawval=False, normalize_flux=astro.normalize_flux, 
                                                         normalize_total_flux=astro.normalize_total_flux, 
                                                         normalize_mean_flux=astro.normalize_mean_flux,
                                                         )
            
        y_real = torch.Tensor(y_real).to(astro.device)
    else:
        y_test = torch.Tensor(y_test).to(astro.device)
        y_real = torch.Tensor(y_real).to(astro.device)
        
        
    with torch.no_grad():
        features_test = model.cond_net.features(y_test).detach().cpu().numpy()
       
        features_real = model.cond_net.features(y_real).detach().cpu().numpy()
    
    features = np.concatenate([features_test, features_real], axis=0)
    labels = np.array([0]*len(features_test) + [1]*len(features_real))
    
    if do_pca:
        pca = PCA(n_components=n_components_pca)
        features = pca.fit_transform(features)

    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    features_2d = tsne.fit_transform(features)
    
    if cval_name is not None: # will use param as color
        param_test = astro.x_to_params(x_test)
        index = astro.x_names.index(cval_name)
        cval = param_test[:, index]
        

    return features_2d, labels, cval


def calculate_umap(model, astro, smoothing=False, do_pca = False, n_components_pca = 50,
                   n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42, 
                   cval_name = None,
                   y_test=None, y_real=None, x_test=None,
                   ):
    
    import umap
    from sklearn.decomposition import PCA
    
    if y_test is None or y_real is None or x_test is None:
        x_test, y_test = load_data_for_situation(astro, smoothing=smoothing)
        # rescaled parameters. y include all thing
        y_test = torch.Tensor(y_test).to(astro.device)
        
        # Load real data
        if astro.prenoise_training:
            y_real, s_real =  astro.get_real_data(rawval=False, normalize_flux=astro.normalize_flux, 
                                                         normalize_total_flux=astro.normalize_total_flux, 
                                                         normalize_mean_flux=astro.normalize_mean_flux,
                                                  )
            y_real = np.hstack( [y_real, s_real] )
        
        else:
            y_real =  astro.get_real_data(rawval=False, normalize_flux=astro.normalize_flux, 
                                                         normalize_total_flux=astro.normalize_total_flux, 
                                                         normalize_mean_flux=astro.normalize_mean_flux,
                                                         )
            
        y_real = torch.Tensor(y_real).to(astro.device)
    else:
        y_test = torch.Tensor(y_test).to(astro.device)
        y_real = torch.Tensor(y_real).to(astro.device)
        
        
    with torch.no_grad():
        features_test = model.cond_net.features(y_test).detach().cpu().numpy()
       
        features_real = model.cond_net.features(y_real).detach().cpu().numpy()
    
    features = np.concatenate([features_test, features_real], axis=0)
    labels = np.array([0]*len(features_test) + [1]*len(features_real))
    
    if do_pca:
        pca = PCA(n_components=n_components_pca)
        features = pca.fit_transform(features)

    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist,metric=metric, random_state=random_state)
    features_2d = reducer.fit_transform(features)
    
    if cval_name is not None: # will use param as color
        param_test = astro.x_to_params(x_test)
        if cval_name =='Teff':
            index = astro.x_names.index('logTeff')
            cval = 10**(param_test[:, index])
        else:
            index = astro.x_names.index(cval_name)
            cval = param_test[:, index]
        

    return features_2d, labels, cval

def visualize_domain(features_2d, labels, 
                     color_source = 'tab:blue', norm=None, cmap = plt.get_cmap("gnuplot"), clabel=None, marker_source = 'o', marker_target='s',
                     label_source='Source', label_target='Target', alpha_source=0.2, alpha_target=0.7, size_source=8, size_target=5, color_target = 'tab:green', 
                     fig=None, ax=None, figsize=[6,5], legend_loc='best', legend_size = 'medium', plot_colorbar=False,
                     title = 't-SNE of Source and Target Features', title_size='large', clabel_size = 'large', 
                     xlabel=None, ylabel=None, xylabel_size = 'large', xytickoff=False, xyticklabelsize='small',
                     return_figure = False, figname=None, 
                    
                    ):
    # labels 0: source (simulation), 1: target (real)
    if fig is None and ax is None:
        fig, ax = plt.subplots(1,1, figsize=figsize, tight_layout=1)
    if type(color_source) == str: # just one color
        cval = color_source
    else:
        cval = color_source
        plot_colorbar = True
        if norm is None: norm = clr.Normalize(vmin=cval.min(), vmax=cval.max())
        # cmap = copy.copy(cmap)
            
    im=ax.scatter(features_2d[labels == 0, 0], features_2d[labels == 0, 1], marker=marker_source, label=label_source, alpha=alpha_source, s=size_source, c=cval, norm=norm, cmap=cmap)

    ax.scatter(features_2d[labels == 1, 0], features_2d[labels == 1, 1], marker=marker_target, label=label_target,
               alpha=alpha_target, s=size_target, c=color_target, norm=norm, cmap=cmap, edgecolors='w',)
    ax.legend(loc=legend_loc, fontsize=legend_size)
    ax.set_title(title, size=title_size)
    if xlabel is not None: ax.set_xlabel(xlabel, size=xylabel_size)
    if ylabel is not None: ax.set_ylabel(ylabel, size=xylabel_size)
    if xytickoff:
        ax.tick_params(axis='both', labelleft=False, labelbottom=False)
    else:
        ax.tick_params(axis='both', labelsize=xyticklabelsize)
    

    if plot_colorbar:
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes("right",size="5%", pad="2%") 
        cbar=fig.colorbar(im, cax=cbar_ax,orientation='vertical')
        # cbar_ax.set_title(clabel, size=clabel_size)
        cbar_ax.set_ylabel(clabel, size=clabel_size)

    if figname is not None:
        fig.savefig(figname, dpi=300)
        plt.close()


    if return_figure:
        return (fig, ax)
    
def calculate_domain_distance(model, astro, smoothing=False, option='CMD',
                              
                          rbf_sigma = 1.0, rbf_sigma_list = [1, 2, 5, 10, 20, 40], 
                          y_test=None, y_real=None, ):
    
    if np.logical_or(y_test is None, y_real is None):
    
        x_test, y_test = load_data_for_situation(astro, smoothing=smoothing)
        # rescaled parameters. y include all thing
        y_test = torch.Tensor(y_test).to(astro.device)
        
        # Load real data
        if astro.prenoise_training:
            y_real, s_real =  astro.get_real_data(rawval=False, normalize_flux=astro.normalize_flux, 
                                                         normalize_total_flux=astro.normalize_total_flux, 
                                                         normalize_mean_flux=astro.normalize_mean_flux, )
            y_real = np.hstack( [y_real, s_real] )
        
        else:
            y_real =  astro.get_real_data(rawval=False, normalize_flux=astro.normalize_flux, 
                                                         normalize_total_flux=astro.normalize_total_flux, 
                                                         normalize_mean_flux=astro.normalize_mean_flux, )
        y_real = torch.Tensor(y_real).to(astro.device)
    else:
        y_test = torch.Tensor(y_test).to(astro.device)
        y_real = torch.Tensor(y_real).to(astro.device)
    
    with torch.no_grad():
        features_test = model.cond_net.features(y_test)
        features_real = model.cond_net.features(y_real)
        


    def compute_cmd(x, y, n_moments=5):
        """
        x, y: NumPy arrays of shape (N, D)
        n_moments: int, number of central moments to include (default: 5)
    
        Returns:
            scalar float: CMD distance
        """
        assert x.shape[1] == y.shape[1], "Feature dimension mismatch"
    
        mx = x.mean(axis=0)
        my = y.mean(axis=0)
    
        sx = x - mx
        sy = y - my
    
        cmd = np.linalg.norm(mx - my, ord=2)
    
        for i in range(2, n_moments + 1):
            moment_x = np.mean(sx ** i, axis=0)
            moment_y = np.mean(sy ** i, axis=0)
            cmd += np.linalg.norm(moment_x - moment_y, ord=2)
    
        return cmd

        
    
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
    
  

    def compute_mmd_mk_rbf(x, y, sigmas=[1, 5, 10]):
        """
        x: (N, D) tensor (source or fake features)
        y: (M, D) tensor (target or real features)
        sigmas: list of floats, kernel bandwidths for RBF
    
        Returns: scalar tensor - MK-MMD^2
        """
        def rbf_kernel_multi(a, b, sigmas):
            dists = torch.cdist(a, b, p=2).pow(2).unsqueeze(0)  # (1, N, M)
            sigmas = torch.tensor(sigmas, device=a.device).view(-1, 1, 1)  # (S, 1, 1)
            kernels = torch.exp(-dists / (2 * sigmas ** 2))  # (S, N, M)
            return kernels.mean(0)  # average over S → (N, M)
    
        K_xx = rbf_kernel_multi(x, x, sigmas)
        K_yy = rbf_kernel_multi(y, y, sigmas)
        K_xy = rbf_kernel_multi(x, y, sigmas)
    
        mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
        return mmd
    
    def compute_mmd_rbf_blockwise(x, y, sigma=1.0, block_size=128):
        def rbf_kernel_blockwise(a, b):
            total = 0.0
            count = 0
            for i in range(0, a.size(0), block_size):
                a_block = a[i:i+block_size]
                for j in range(0, b.size(0), block_size):
                    b_block = b[j:j+block_size]
                    dist2 = torch.cdist(a_block, b_block, p=2).pow(2)
                    k = torch.exp(-dist2 / (2 * sigma ** 2))
                    total += k.sum()
                    count += k.numel()
            return total / count

        K_xx = rbf_kernel_blockwise(x, x)
        K_yy = rbf_kernel_blockwise(y, y)
        K_xy = rbf_kernel_blockwise(x, y)
    
        return K_xx + K_yy - 2 * K_xy


    if option=='CMD':
        cmd_val = compute_cmd(features_test.detach().cpu().numpy(), features_real.detach().cpu().numpy(), n_moments=5)
        return cmd_val
    if option=='MK-MMD':
        mmd_val = compute_mmd_mk_rbf(features_test, features_real, sigmas=rbf_sigma_list)
        return mmd_val.item()
    if option=='B-MMD':
        mmd_val = compute_mmd_rbf_blockwise(features_test, features_real, sigma=rbf_sigma, block_size=128)
        return mmd_val.item()
    
    # mmd_val = compute_mmd_rbf(features_test, features_real, sigma=rbf_sigma)
    
    # print(f"MMD: {mmd_val.item():.6f}")
    # return mmd_val.item()

    
    
def eval_DA_status(model, astro, save_features=True, tf_filename=None, rf_filename=None, Ddist_kwarg=None, tSNE_kwarg=None, run_tSNE=True, run_Ddist=True):
    # convinent function group to run during training
    previous_filter = warnings.filters[:]
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # To save time, load data in advance and give them as keywords
    x_test, y_test = load_data_for_situation(astro, smoothing=False)
    # rescaled parameters. y include all thing
    
    # Load real data
    if astro.prenoise_training:
        y_real, s_real =  astro.get_real_data(rawval=False, normalize_flux=astro.normalize_flux, 
                                                     normalize_total_flux=astro.normalize_total_flux, 
                                                     normalize_mean_flux=astro.normalize_mean_flux,)
        y_real = np.hstack( [y_real, s_real] )
    
    else:
        y_real =  astro.get_real_data(rawval=False, normalize_flux=astro.normalize_flux, 
                                                     normalize_total_flux=astro.normalize_total_flux, 
                                                     normalize_mean_flux=astro.normalize_mean_flux,  )
        
       
    if save_features:
        # calculate features
        
        with torch.no_grad():
            features_test = model.cond_net.features(torch.Tensor(y_test).to(astro.device)).detach().cpu().numpy()
            features_real = model.cond_net.features(torch.Tensor(y_real).to(astro.device)).detach().cpu().numpy()
        # change to float32. save in .npy
        # save test features 
        np.save(tf_filename, features_test.astype(np.float32))
        # save real features
        np.save(rf_filename, features_real.astype(np.float32))
        
    if run_Ddist:
        D_test, D_real = calculate_D(model, astro, smoothing=astro.train_smoothing,
                                     y_test=y_test, y_real=y_real)
        
        plot_D_distribution(D_test, D_real, return_figure=False, **Ddist_kwarg, )
    
    if run_tSNE:
        cval_name = astro.x_names[0]
        cval_name = 'Teff'
        # features_2d, labels, cval = calculate_tSNE(model, astro, smoothing=astro.train_smoothing, 
        #                    do_pca = False, n_components_pca = 50, cval_name = cval_name,
        #                     n_components=2, perplexity=30, random_state=42,
        #                     y_test=y_test, y_real=y_real, x_test=x_test)
        
        features_2d, labels, cval = calculate_umap(model, astro, smoothing=astro.train_smoothing, 
                           do_pca = False, cval_name = cval_name,
                           n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42, 
                           y_test=y_test, y_real=y_real, x_test=x_test)
                           
        try:
            clabel=astro.exp.title_unit_dic[cval_name]
        except:
            clabel = cval_name
            
        if cval_name=='logTeff':
            color_target = np.log10((pandas.read_csv(astro.real_database))['Teff'])
        elif cval_name=='Teff':
            color_target = (pandas.read_csv(astro.real_database))['Teff']
        else:
            color_target='tab:green'
        
        visualize_domain(features_2d, labels, 
                         color_source=cval, clabel = clabel, color_target=color_target,
                         return_figure = False, **tSNE_kwarg)
                            
    warnings.filters = previous_filter
    

    
    

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
    # header += ['L_rev_train_mn', 'L_rev_test_mn', 'L_rev_train_mdn', 'L_rev_test_mdn']
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
               
        loss_data = pandas.read_csv(info_dic['loss'], sep='\t', usecols=header[:-1]).values[-1, :]
        loss_data = np.append(loss_data, [training_status])
               
    except:
        print("Caanot find loss file (%s)"%info_dic['loss'])
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
    
    
    
    
    
    
    
