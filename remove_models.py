#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 16:25:09 2022

@author: daeun

2023. 1. 24

Remove unnecessary model .pt files

Necessary input argument:
- hyperparameter directory (the code remove ~dir/model*/*.pt)
- excluding model numbers
"""
import os, sys, glob
from argparse import ArgumentParser

parser = ArgumentParser()
# parser.add_argument("-c", "--config", dest="config_file", default='./config.py',
#                     help="Run with specified config file as basis.")

parser.add_argument('-o', '--outputdir', required=True, help="Hyperparameter search directory")
parser.add_argument('-e','--exclude', required=True, type=int, nargs='+', help="Excluding numbers") # python ~.py -o directory -e number1 number 2 number 3 number 4


# Import default config
args = parser.parse_args()

hps_dir = args.outputdir
exclude_numbers = args.exclude # list of integer


print(exclude_numbers)

if not os.path.exists(hps_dir):
    sys.exit('There is no %s \nCheck the directory name'%hps_dir)

# find model directories

model_path_list = sorted(glob.glob(hps_dir+"model_*/"))
model_path_list = [k.replace(hps_dir,'') for k in model_path_list]
# print(model_path_list)
_ = model_path_list[0].split('_')
suffix = model_path_list[0].replace('_'+_[-1],"")
print("network suffix: %s"%suffix)

for model_path in model_path_list:
    
    _ = model_path.split('_')
    num = int(_[-1][:-1])
    if num in exclude_numbers:
        continue
    else:
        files = glob.glob(hps_dir + model_path+'*.pt')
        if len(files)>0:
            for file in files:
                print("rm %s"%file)
                os.system("rm %s"%file)
        
#     sym_name = os.path.basename(ori_path)
#     print(ori_path, sym_name)
    
#     # already done?
#     if os.path.exists(proj_dir + sym_name):
#         print('%s dir already exists in %s (pass)'%(sym_name, proj_dir))
#         n_pass += 1
#     else:
#         # file really exist?
#         if os.path.exists(ori_path):
#             # print('ln -s {} {}'.format(ori_path, proj_dir+sym_name   )  )
#             os.system('ln -s {} {}'.format(ori_path, proj_dir+sym_name   )  )
#             print('{} done'.format(sym_name))
#             n_done +=1
            
#         else:
#             print('%s does not exist(fail)'%(ori_path))
#             n_fail += 1
            
            
# print('Finished')
# print('done: %d \tpass: %d \tfail: %d'%(n_done, n_pass, n_fail))