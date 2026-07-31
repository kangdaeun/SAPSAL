#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 15:08:54 2026

@author: daeun

Functions to easily load the pre-trained network.
Information on file names code names, etc
"""
from importlib import resources
import os
import zipfile
import numpy as np

from .cINN_config import read_config_from_file
from .expander import convert_temp_to_sptnum, convert_temp_to_spt

AVAILABLE_NET_CODES = ['v1_settl', 'v2_k25', 'v3_vis', 'v3_uv']

NETWORK_SHORT_NAMES = {
    'v1_settl': 'Settl-Net',
    'v2_k25': "K25-Net",
    'v3_vis': "Vis-Net",
    'v3_uv': "UV-Net",
}
NETWORK_OFFICIAL_NAMES = {
    'v1_settl': 'SAPSAL-v1-Settl',
    'v2_k25': "SAPSAL-v2-K25",
    'v3_vis': "SAPSAL-v3-Vis",
    'v3_uv': "SAPSAL-v3-UV",
}

CONFIG_FILES = {
    'v1_settl': 'v1_settl/config_v1_Settl.py',
    'v2_k25': "v2_k25/config_v2_K25.py",
    'v3_vis': "v3_vis/config_v3_Vis.py",
    'v3_uv': "v3_uv/config_v3_UV.py"
    }
# key: network_code, value: config file path in sapsal/networks/

NETWORK_FILES = {
    'v1_settl': 'v1_settl/v1_Settl_Net.pt',
    'v2_k25': "v2_k25/v2_K25_Net.pt",
    'v3_vis': "v3_vis/v3_Vis_Net.pt",
    'v3_uv': "v3_uv/v3_UV_Net.pt"
    }

NETWORK_TRAINING_RANGES = {
    'v1_settl': {
                'Teff':(2600, 7000),  'logTeff':(np.log10(2600), np.log10(7000)),
                'logG': (2.5, 5.0),
                'A_V': (0, 10),   
                },

    'v2_k25': {
                'Teff':(2600, 7000),  'logTeff':(np.log10(2600), np.log10(7000)),
                'logG': (2.5, 5.0),
                'A_V': (0, 10),
                'veil_r': (0,2),
                'library': (0, 1), 
            },
    'v3_vis': {
                'Teff':(2330, 7000),  'logTeff':(np.log10(2330), np.log10(7000)),
                'logG': (2.5, 5.5),
                'A_V': (0, 10),
                'log_veil_r': (-4, 1), 'veil_r': (10**-4, 10**1),
                'R_V': (2, 6),
                'library': (-1, 1),  # -1: FRAPPE, 0: Settl, 1: Dusty
                'Tslab': (5000, 11000),
                'log_ne': (11, 16),
                'log_tau0':(np.log10(0.01), np.log10(5)),
                'log_Fslab':(3.9, 4.32),
                # 'SpTind': (convert_temp_to_sptnum(2330), convert_temp_to_sptnum(7000)), # SpTind of 2330 is larger than that of 7000K
                # 'SpT': (convert_temp_to_spt(2330, option='Tpl')[:4], convert_temp_to_spt(7000, option='Tpl')[:4]),
                },
    'v3_uv': {
                'Teff':(2330, 7000),  'logTeff':(np.log10(2330), np.log10(7000)),
                'logG': (2.5, 5.5),
                'A_V': (0, 10),
                'log_veil_r': (-4, 1), 'veil_r': (10**-4, 10**1),
                'R_V': (2, 6),
                'library': (-1, 1),  # -1: FRAPPE, 0: Settl, 1: Dusty
                'Tslab': (5000, 11000),
                'log_ne': (11, 16),
                'log_tau0':(np.log10(0.01), np.log10(5)),
                'log_Fslab':(3.9, 4.32),
            }         
    }   



# 일단 config 절대 혹은 상대 경로를 정확하게 .
# config와 같은 폴더에 있으니 그 폴더가 proj_dir가 된다.
# 편의상 tablename은 지워버리는 것이 문제가 없을까?
# 한 함수내에 경로를 얻고 네트워크를 로드해서 config만 전달하는 편이 나을까?
# 일단 별도 함수로

def load_pretrained_network(net_code, remove_table_paths=True, verbose=True):
    # make path for config
    config_rel_path = CONFIG_FILES.get(net_code, None)
    if config_rel_path is None:
        raise ValueError(f"Network code '{net_code}' not recognized. Available codes: {AVAILABLE_NET_CODES}")
    
    
    model_rel_path = NETWORK_FILES.get(net_code, None)
    if model_rel_path is None:
        raise ValueError(f"Network code '{net_code}' not recognized. Available codes: {AVAILABLE_NET_CODES}")
    
    if verbose:
        print("Requested network:", NETWORK_SHORT_NAMES.get(net_code, 'Unknown'), f"(code: {net_code}, Official Name: {NETWORK_OFFICIAL_NAMES.get(net_code, 'Unknown')})")
    
    # Check if the corresponding network file exists. or in zip file.
    net_source = resources.files("sapsal.networks").joinpath(model_rel_path)
    with resources.as_file(net_source) as pt_path:
        # 3. .pt 파일이 이미 존재하는지 확인
        if not pt_path.exists():  
            # 4. .pt가 없다면 .zip 파일 경로 확인
            zip_path = str(pt_path) + ".zip"
            if os.path.exists(zip_path):
                if verbose:
                    print(f"Unzip the network ({zip_path})... (Only once per network)")
                # 5. 압축 해제 (해당 파일이 있는 부모 폴더에 해제)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # 추출 위치는 .pt 파일이 있어야 할 부모 폴더
                    zip_ref.extractall(os.path.dirname(zip_path))

    # Path for config file
    source = resources.files("sapsal.networks").joinpath(config_rel_path)
    # if verbose:
    #     print(f"Loading network config...")


    with resources.as_file(source) as safe_path:
        config_file_path = str(safe_path)
        config_dir_path = str(safe_path.parent) + os.sep

        config = read_config_from_file(config_file_path, proj_dir=config_dir_path)
        if remove_table_paths:
            # if verbose:
            #     print("(Removing paths of training data from config: currently, training data is not supported in SAPSAL github. If you want to keep the paths, set remove_table_paths=False.)")
            if config.tablename is not None:
                config.tablename = None
            if config.slab_grid is not None:
                config.slab_grid = None

        if verbose:
            print("Network config loaded successfully.")
        
        return config