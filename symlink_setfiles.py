#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 16:25:09 2022

@author: daeun

2022. 8. 30
In proj_dir, you need ess_files/ and Database/
cINN_set is not necessary but it's convinient to make cINN_set as well (for some codes in ess_files, it is necessary)

Input argument of this code is your proj_dir (relative path to current working directory or abspath)

The code will make a symbolic link of
ess_files/ 
Database/
cINN_set/
in your proj_dir
"""
import os, sys, glob

# ess_dir = '../Networks/ess_files/' # real files
# db_dir = '../Database/' # real files

ess_dir = '/export/scratch/dekang/ECOGAL/cinn_ssp/Networks/ess_files' # abspath to real files (not symbolic link)
db_dir = '/export/scratch/dekang/ECOGAL/cinn_ssp/Database' # abspath
cINN_set_dir = '/export/scratch/dekang/ECOGAL/cinn_ssp/Networks/cINN_set'


proj_dir = sys.argv[1]
if proj_dir[-1]!='/':
    proj_dir = proj_dir+'/'
    
if not os.path.exists(proj_dir):
    sys.exit('There is no %s \nCheck the directory name'%proj_dir)

ori_path_list = [ess_dir, db_dir, cINN_set_dir]
    
n_pass = 0
n_done = 0
n_fail = 0

for ori_path in ori_path_list:
    
    sym_name = os.path.basename(ori_path)
    print(ori_path, sym_name)
    
    # already done?
    if os.path.exists(proj_dir + sym_name):
        print('%s dir already exists in %s (pass)'%(sym_name, proj_dir))
        n_pass += 1
    else:
        # file really exist?
        if os.path.exists(ori_path):
            # print('ln -s {} {}'.format(ori_path, proj_dir+sym_name   )  )
            os.system('ln -s {} {}'.format(ori_path, proj_dir+sym_name   )  )
            print('{} done'.format(sym_name))
            n_done +=1
            
        else:
            print('%s does not exist(fail)'%(ori_path))
            n_fail += 1
            
            
print('Finished')
print('done: %d \tpass: %d \tfail: %d'%(n_done, n_pass, n_fail))