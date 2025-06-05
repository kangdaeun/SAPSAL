#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 14:15:14 2022

@author: daeun

Validation with Class III template stars

"""
import numpy as np
import matplotlib.pyplot as plt
import os,sys, glob
from astropy.io import ascii
# from time  import time
# import matplotlib.colors as clr
# import copy
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from argparse import ArgumentParser
import warnings
# from pathlib import Path
from astropy.table import Table#, hstack
warnings.filterwarnings("ignore")
sys.path.append("/export/scratch/dekang/ECOGAL/cinn_ssp/Networks/")
from sapsal.cINN_config import read_config_from_file
from sapsal.tools.test_tools import check_training_status


main_dir = '/export/scratch/dekang/ECOGAL/cinn_ssp/'

### parameter ranges of training data
db_range_dic = {}
db_range_dic['NextGen']={
    'Teff':[2600, 7000],
    'logTeff':[np.log10(2600), np.log10(7000)],
    'logG': [2.5, 5],
}
db_range_dic['Dust']={
    'Teff':[2600, 4000],
    'logTeff':[np.log10(2600), np.log10(4000)],
    'logG': [3, 5],
}
db_range_dic['Settl']={
    'Teff':[2600, 7000],
    'logTeff':[np.log10(2600), np.log10(7000)],
    'logG': [2.5, 5],
}
db_range_dic['SpD']={
    'Teff':[2600, 7000],
    'logTeff':[np.log10(2600), np.log10(7000)],
    'logG': [2.5, 5],
}

### Shared figure keywards
figure_kwarg = {
    'xlabelsize':'large', 'ylabelsize':'large', 'titlesize':'x-large',
    'yticklabelsize':'medium', 'xticklabelsize':'medium', 'txtsize':'small',
    'cmap':cm.get_cmap("RdYlBu"), 'cbar_ticklabelsize':'small', 
    'sval':38, 'annsize':'small',
}
txtbox_kwarg = { 
    'bbox':dict(boxstyle='round', facecolor='w', alpha=0.6, edgecolor='silver')
}

scale_dic={
    'dA':(-0.5,2.5),
    # 'dT':(-0.1,0.1),
    'dT':(-250,250),
    'tT':(2700, 5500),
}
range_dic={
    'tT1':[2000, 7150], 'tT2':[2000, 6150],
    'dA': [-1, 5], 'dT': [-360, 460]
}
label_dic = {
    'tT': 'T$_{\mathrm{eff}}^{\mathrm{True}}$ [K]',
    'dA': '$\Delta$ Av (MAP-True)',
    'dT': '$\Delta \mathrm{T}_{\mathrm{eff}}$ (MAP-True) [K]',
    'mT': 'T$_{\mathrm{eff}}^{\mathrm{MAP}}$ [K]',
}

# posterior setting
N_pred = 4096
map_kwarg = {'bw_method':'silverman', 
             'n_grid':1024, 
             'use_percentile': None,
             'plot':False }


def scatter_color(fig, ax, xval, yval, cval, sval=40, plot_cbar=True,
                     
                    marker='o', edgecolors='grey', cmap = cm.get_cmap("RdYlBu"), vmin=None, vmax=None,          
                    xrange=None, yrange=None,
                    xlabel=None, ylabel=None, title=None, 
                    xlabelsize='large', ylabelsize='large', titlesize='x-large',
                    xticklabelsize='medium', yticklabelsize='medium', txtsize = 'small',
                    cbar_position='right', cbar_size="5%", cbar_pad="1%", cbar_ticklabelsize='small',
                    cbar_title=None, cbar_titlesize='small', cbar_Ntick=3, cbar_title_rotation=None,
                   **kwarg,
   
                 ):
    
    if vmin is None:
        vmin = np.nanmin(cval)
    if vmax is None:
        vmax = np.nanmax(cval)
    
    ims=ax.scatter(xval, yval, marker=marker, c=cval, s=sval, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors=edgecolors)
    
    if xrange: ax.set_xlim(xrange)
    if yrange: ax.set_ylim(yrange)
    if title: ax.set_title(title, size=titlesize)
    if xlabel: ax.set_xlabel(xlabel, size=xlabelsize)
    if ylabel: ax.set_ylabel(ylabel, size=ylabelsize)
    
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
        





#%%
##########
## MAIN ##
##########

if __name__=='__main__':

    # Parse optional command line arguments
    parser = ArgumentParser()
    # parser.add_argument("-c", "--config", dest="config_file", default='./config.py',
    #                     help="Run with specified config file as basis.")
    
    parser.add_argument('-c', '--config_file', required=False, default=None, help="config file")
    parser.add_argument('-cd', '--config_dir', required=False, default=None, help="Dirpath to config files")
    parser.add_argument('-pd', '--proj_dir', required=True, help="project directory where we can find output files" )
    parser.add_argument('-rn', '--renew', required=False, default=False, help="Renewal of existing results")
    # parser.add_argument('-s','--suffix', required=False, default=None, help="Output suffix")
    # parser.add_argument('-o','--outputdir', required=False, default=None, help="Output directory")
    # parser.add_argument('-r','--resume', required=False, default=None, help="Resume Hyperparameter search or not (T/F)")
    
    # Import default config
    args = parser.parse_args()
    if args.renew == "True" or args.renew=="1":
        renewal = True
        print("Renew and overwrite the existing results")
    else:
        renewal = False
    
    cINN_dir = args.proj_dir
    
    # ================ Read config files ====================
    config_list = []
    config_status = []
    if args.config_dir is not None: # run multiple networks
        # find all config files in the path
        config_file_list = glob.glob(args.config_dir+'c_*.py')
        if len(config_file_list)==0:
            sys.exit("No config files in config_dir (%s)"%args.config_dir)
        else:
            config_file_list = sorted(config_file_list)
            print("%d config files detected"%(len(config_file_list)))
            for file in config_file_list:
                c = read_config_from_file(file, proj_dir=cINN_dir, verbose=True)
                if check_training_status(c)==1.0:
                    config_status.append(True)
                else:
                    config_status.append(False)
                config_list.append(c)
                
    elif args.config_file is not None: # run for only one network
        c = read_config_from_file(args.config_file, proj_dir=cINN_dir, verbose=True)
        config_list.append(c)
        if check_training_status(c)==1.0:
            config_status.append(True)
        else:
            config_status.append(False)    
    else:
        sys.exit("No config file or config_dir specified")
     
    print("%d Networks to run"%(len(config_list)))
    exp = c.import_expander()
    device = exp.find_gpu_available(return_str=True)
    
    # Save data (median abs (dval) without outliers)
    mdn_dT = []
    mdn_dA = []
    mdn_tot_err = [] # root( dlogT^2 + dA^2)
    
    c_id = []
    
    verbose = True
    #==================== START =====================
    for i_config, c in enumerate(config_list):
        
        
        c.device = device
        c.print_short_setting()
        network_name = os.path.basename(c.filename).replace('.pt','')
        
        try:
            a=int(network_name.split('_')[-1])
            c_id.append(a)
            net_code = '_'.join(network_name.split('_')[:-1])
        except:
            net_code = network_name
        print("Net code:",net_code)
        
        # pass if network is not good
        if config_status[i_config] is False:
            print("Pass %s"%network_name)
            mdn_dT.append(np.nan); mdn_dA.append(np.nan); mdn_tot_err.append(np.nan)
            continue
        
        if 'Stl' in c.filename: db_code = 'Settl'
        elif 'NG' in c.filename: db_code = 'NextGen'
        elif 'Dust' in c.filename: db_code = 'Dust'
        elif 'SpD' in c.filename: db_code = 'SpD'
        db_range = db_range_dic[db_code]
        if 'TGA' in c.filename:
            if 'mA' in c.filename: db_range['A_V'] = [-2,10]
            else: db_range['A_V'] = [0,10]
        print("DB range (%s):"%db_code, db_range)
        
        
        #===================== READ SPECTRA AND CATALOG ==============================
        obs_dir = main_dir + 'Results/Template_spectra/respectrallibraryforcinnfistversionavailable/'
        sfiles = sorted(glob.glob(obs_dir + '*.dat'))

        if 'nHa' in c.filename:
            filter_Ha = True
        else: 
            filter_Ha = False
        if 'res' in c.filename:
            filter_red = True
        else:
            filter_red = False
        # filter_Ha = True # 692-720
        
        # ===================== Read Catalog =============================
        # cat = ascii.read(obs_dir+'summary_classIII.txt') # old one
        cat = ascii.read(obs_dir+'ClassIII-templates_230216.txt')
        # print(len(cat))
        
        # ================= Match caglog and existing spec files ==================
        index = []
        for file in sfiles.copy():
            roi_true = np.where(cat['NAME']==os.path.basename(file).split('VIS')[0][:-1])[0]
            if len(roi_true)==1:
                index.append(roi_true[0])
            else:
                # no informatino in cat, remove in sfiles
                sfiles.remove(file)
                if i_config==0:
                    print("Exclude %s"%file)
        #     else:
        #         print("?")
        if i_config==0:
            print("Number of spectra matched: %d"%len(sfiles))
        
        cat = cat[np.array(index)]
        cat['logTeff'] = np.log10(cat['Teff'])
        
        #================= Sort catalog and sfiles =================
        index = np.argsort(cat['Teff'])
        cat = cat[index]; sfiles = np.array(sfiles)[index]
        #================== Read spectra =====================
        spec_list = []
        for i_file, file in enumerate(sfiles):
            a = np.loadtxt(file)
            spec = a[:-1]
            if filter_Ha:
                spec = np.append( spec[:692], spec[721:])
            elif filter_red:
                nn = len(c.y_names)
                spec = a[:nn]
 
            spec_list.append(spec)
        
            
        
        if i_config==0:  
            print("Read %d template spectra"%len(spec_list))

        print("Check dim(y) of spectra and network")
        print("spec:",len(spec_list[0]), "net:",len(c.y_names))
        if len(spec_list[0])!=len(c.y_names):
            sys.exit("Dimension not matched!")
        
           
        
        plotdir = main_dir + 'Results/Template_results/Network_Generalization/'+net_code+'/' 
        if not os.path.exists(plotdir):
            os.system('mkdir -p {}'.format(plotdir))
            
        postdir = plotdir+'result/'
        if not os.path.exists(postdir):
            os.system('mkdir -p {}'.format(postdir))
            
            
        #================ Run posterior =========================
        post_result_filename = postdir+'%s_result.dat'%network_name
        if os.path.exists(post_result_filename) * (renewal==False):
            print("Read result file")
            res = ascii.read(post_result_filename,  delimiter='\t', format='commented_header')
            ind = []
            for i in range(len(cat)):
                ind.append( np.where(res['NAME']==cat['NAME'][i])[0][0])
            ind = np.array(ind)
            res = res[ind]
            
            map_names = [param for param in res.colnames if '_MAP' in param]
            unc_names = [param for param in res.colnames if '_u68' in param]
            
            map_table = Table()
            unc_table = Table()
            for col in map_names:
                map_table[col.replace('_MAP','')] = res[col]
            for col in unc_names:
                unc_table[col.replace('_u68','')] = res[col]
                
            # print(map_table)
           
        else:     
            obs_array = np.array(spec_list)
            
            #========== Normalize spectra if network used normalization ==========
            if c.normalize_flux:
                obs_array = exp.normalize_flux(obs_array, normalize_total_flux=c.normalize_total_flux, 
                                   normalize_mean_flux=c.normalize_mean_flux)
            
            post_list = exp.get_posterior(obs_array, c, N=N_pred, use_group=True,
                                          group=len(obs_array))
            
            unc_list = []
            map_list = []
            for i_post, post in enumerate(post_list):
                unc = exp.calculate_uncertainty(post, c, add_Teff=True)
                maps = exp.calculate_map(post, c, **map_kwarg)
                unc_list.append(unc)
                map_list.append(maps)
            unc_list = np.array(unc_list)
            map_list = np.array(map_list)
            
            map_table = Table(map_list, names=c.x_names)
            map_table['Teff'] = 10**map_table['logTeff']
            unc_table = Table(unc_list, names=c.x_names+['Teff'])
            
            
            # Make final result file: Combine catalog + map + u68
            res = cat.copy()
            formats={}
            for param in map_table.colnames:
                res[param+'_MAP'] = map_table[param]
                formats[param+'_MAP'] = '%#.8g'
            for param in unc_table.colnames:
                res[param+'_u68'] = unc_table[param]  
                formats[param+'_u68'] = '%#.8g'
        
            ascii.write(res, post_result_filename, delimiter='\t', format='commented_header',
                        overwrite=True, formats=formats)

       # ==================== Plot figure =========================== 
        fig, axis = plt.subplots(2,2,figsize=[8,8], tight_layout=True)
    
        #1) tT vs mT (dT)
        ax = axis[0,0]
        roi_tout = np.logical_or((cat['Teff'] < db_range['Teff'][0]),(cat['Teff'] > db_range['Teff'][1]))
        yval = map_table['Teff']
        xval = cat['Teff']
        dval = yval - xval; 
        med = np.median(abs(dval)); med_wo=np.median(abs(dval[np.invert(roi_tout)]))
        
        vmin, vmax = scale_dic['dT']
        xrange =  range_dic['tT1']; yrange=range_dic['tT1']
        
        txt = []
        txt.append('Median(|dT|)={:.4g} (All:{:d})'.format(med, len(dval)))
        txt.append('Median(|dT|)={:.4g} (in DB:{:d})'.format(med_wo, len(dval)-np.sum(roi_tout)))
        
        xx = np.linspace(2000, 10000,100)
        ax.plot(xx, xx, ls='--', color='k', lw=1)
        
        scatter_color(fig, ax, xval, yval, dval, xrange=xrange, yrange=yrange,
                      vmin=vmin, vmax=vmax,
                     xlabel=label_dic['tT'], ylabel=label_dic['mT'],
                      cbar_title=label_dic['dT'], **figure_kwarg,
                     )
        
        l1=ax.axvline(x=db_range[xval.name][0], label='DB boundary',ls=':',color='g')
        ax.axvline(x=db_range[xval.name][1], label='DB boundary',ls=':',color='g')
        ax.axhline(y=db_range[xval.name][0], label='DB boundary',ls=':',color='g')
        ax.axhline(y=db_range[yval.name][1], label='DB boundary',ls=':',color='g')
        ax.legend(handles=[l1], loc='upper left', fontsize=figure_kwarg['txtsize'])
        
        ax.text(0.95, 0.03, '\n'.join(txt), transform=ax.transAxes, ha='right',
                **txtbox_kwarg, fontsize=figure_kwarg['txtsize'])
        
        mdn_dT.append(med_wo)
        # mdn_dlogT.append( np.median( abs(map_table['logTeff']-cat['logTeff'])[np.invert(roi_tout)]   ) )
        
        # 2) tT vs dT (dA)
        ax = axis[1,0]
        
        yval = map_table['Teff'] - cat['Teff']
        xval = cat['Teff']
        dval = map_table['A_V'] - 0.; 
        vmin, vmax = scale_dic['dA']
        xrange =  range_dic['tT1']; yrange=range_dic['dT']
        
        scatter_color(fig, ax, xval, yval, dval, xrange=xrange, yrange=yrange,
                      vmin=vmin, vmax=vmax,
                     xlabel=label_dic['tT'], ylabel=label_dic['dT'],
                      cbar_title=label_dic['dA'], **figure_kwarg,
                     )
        l1=ax.axvline(x=db_range[xval.name][0], label='DB boundary',ls=':',color='g')
        ax.axvline(x=db_range[xval.name][1], label='DB boundary',ls=':',color='g')
        ax.axhline(y=0,ls=':',color='k', lw=1)
        
        # 3) tT vs dA (dT)
        ax = axis[0,1]
        roi_tout = np.logical_or((cat['Teff'] < db_range['Teff'][0]),(cat['Teff'] > db_range['Teff'][1]))
        
        yval = map_table['A_V'] - 0.
        xval = cat['Teff']
        dval = map_table['Teff'] - cat['Teff']; 
        vmin, vmax = scale_dic['dT']
        xrange =  range_dic['tT2']; yrange=range_dic['dA']
        med_wo = np.median(abs(yval[np.invert(roi_tout)]))
        txt = []
        txt.append('Median(|dAv|)={:.4g} (All:{:d})'.format(np.median(abs(yval)), len(yval)))
        txt.append('Median(|dAv|)={:.4g} (in DB:{:d})'.format(med_wo, len(yval)-np.sum(roi_tout)))
        
        scatter_color(fig, ax, xval, yval, dval, xrange=xrange, yrange=yrange,
                      vmin=vmin, vmax=vmax,
                     xlabel=label_dic['tT'], ylabel=label_dic['dA'],
                      cbar_title=label_dic['dT'], **figure_kwarg,
                     )
        
        l1=ax.axvline(x=db_range[xval.name][0], label='DB boundary',ls=':',color='g')
        ax.axvline(x=db_range[xval.name][1], label='DB boundary',ls=':',color='g')
        ax.axhline(y=0,ls=':',color='k', lw=1)
        ax.text(0.98, 0.9, '\n'.join(txt), transform=ax.transAxes, ha='right',
                **txtbox_kwarg, fontsize=figure_kwarg['txtsize'])
        
        mdn_dA.append(med_wo)
        
        # 4) dT vs dA (tT)
        ax = axis[1,1]
        yval = map_table['A_V'] - 0.
        xval =  map_table['Teff'] - cat['Teff']; 
        dval = cat['Teff']
        vmin, vmax = scale_dic['tT']
        xrange =  range_dic['dT']; yrange=range_dic['dA']
        
        scatter_color(fig, ax, xval, yval, dval, xrange=xrange, yrange=yrange,
                      vmin=vmin, vmax=vmax,
                     xlabel=label_dic['dT'], ylabel=label_dic['dA'],
                      cbar_title=label_dic['tT'], **figure_kwarg,
                     )
        
        for i in range(len(xval)):
            ax.annotate('%d'%i, (xval[i], yval[i]), color='k' , fontsize=figure_kwarg['annsize'])
        
        ax.axhline(y=0,ls=':',color='k', lw=1)
        ax.axvline(x=0,ls=':',color='k', lw=1)
        
        
        fig.suptitle('Net: %s'%network_name, fontsize='xx-large')
        fig.savefig(plotdir+'Comp_TA_4_%s.png'%network_name, dpi=300)
        print("Saved file of %s"%network_name)
        plt.close()
        
        mdn_tot_err.append( np.median( np.sqrt((map_table['logTeff'] - cat['logTeff'])**2. + (map_table['A_V'])**2. ) ) )
        
    
    # Save data
    if len(config_list)>1:
        data = np.array([list(config_file_list), mdn_dT, mdn_dA, mdn_tot_err]).transpose()
        table = Table(data, names=['config_file', 'mdn_dT', 'mdn_dA', 'mdn_tot_err'])
        c_id = np.array(c_id)
        table = table[np.argsort(c_id)]

        ascii.write(table, plotdir+'result.dat', format='commented_header', delimiter='\t', overwrite=True)
        print("Saved collected result")
    
