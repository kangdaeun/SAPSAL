import numpy as np
import matplotlib.pyplot as plt

import torch
import sys
import matplotlib.cm as cm
from time import time

from mpl_toolkits.axes_grid1 import make_axes_locatable
# import matplotlib.gridspec as gridspec

from cINN_set.cINN_config import *
from cINN_set.data_loader import *
#from cINN_set.models import *
from cINN_set.tools import test_tools as tools

GPU_MAX_LOAD = 0.2           # Maximum compute load of GPU allowed in automated selection
GPU_MAX_MEMORY = 0.2         # Maximum memory load of GPU allowed in automated selection
GPU_WAIT_S = 600             # Time (s) to wait between tries to find a free GPU if none was found
GPU_ATTEMPTS = 10            # Number of times to retry finding a GPU if none was found
GPU_EXCLUDE_IDS = [] # List of GPU IDs that are to be ignored when trying to find a free GPU, leave as empty list if none
VERBOSE = True


#config_file = 'config_rncpr02_x7_y2_01'
# config_file = sys.argv[1]
# if config_file[-3:]=='.py':
#     config_file=config_file[:-3]
if __name__ == '__main__':  
    config_file = sys.argv[1]    
    c = read_config_from_file(config_file)
    print(config_file)
    
    astro = DataLoader(c)
    
    if len(sys.argv)==3:
        device = sys.argv[2]
        print('Change device from %s to %s'%(c.device, device))
        c.device = device
    elif 'cuda' in c.device:
        import GPUtil
        DEVICE_ID_LIST = GPUtil.getFirstAvailable(maxLoad=GPU_MAX_LOAD,
                                                  maxMemory=GPU_MAX_MEMORY,
                                                  attempts=GPU_ATTEMPTS,
                                                  interval=GPU_WAIT_S,
                                                  excludeID=GPU_EXCLUDE_IDS,
                                                  verbose=VERBOSE)
        DEVICE_ID = DEVICE_ID_LIST[0]
        c.device = 'cuda:{:d}'.format(DEVICE_ID)
   
    print("Device: ",c.device) 
    astro.device = c.device
    
    c.load_network_model()
        
    z_all = tools.calculate_z(c.network_model, astro, smoothing=c.train_smoothing)
    r1=tools.plot_z(z_all, figname=c.filename+'_z_cov_pdf.pdf', corrlabel=True, legend=True, covariance=True, cmap=cm.get_cmap("gnuplot"), color_letter='r')#, yrange1=[-0.04, 0.6], yrange2=[-0.1, 0.2])
    r2=tools.plot_z(z_all, figname=c.filename+'_z_corr_pdf.pdf', corrlabel=True, legend=True, covariance=False, cmap=cm.get_cmap("gnuplot"), color_letter='r')
    if VERBOSE:
        print("Saved Z covariance and distributions")
    df = tools.latent_normality_tests(z_all, filename=c.filename+'_z_test.csv')
    if VERBOSE:
        print("Saved Z normality tests")
    q1 = tools.qq_plot(z_all, figname=c.filename+'_z_qqplot.pdf')
    if VERBOSE:
        print("Saved Z q-q plot")
