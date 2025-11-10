#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 17:04:05 2025

@author: daeun

Network classes useful for diverse FeatureNet

"""

import torch
import torch.nn as nn
import torch.optim
# from torch.autograd import Variable

from ..FrEIA import framework as Ff
from ..FrEIA import modules as Fm

from typing import List, Tuple, Union, Dict


import numpy as np

class HybridFeatureNet(nn.Module):
    def __init__(self, c, 
                # total_input_length: int,      # 스펙트럼 + Rv의 총 길이 (2001)
                #  num_spectrum_points: int,     # 순수 스펙트럼 플럭스/노이즈 개수 (2000)
                #  num_global_params_per_point: int, # Rv처럼 한 지점에 있는 글로벌 파라미터 개수 (1)
                #  spectrum_cnn_output_dim: int, # 스펙트럼 CNN의 최종 특징 차원
                #  global_param_output_dim: int, # 전역 파라미터 네트워크의 특징 차원
                #  final_output_features: int,   # 최종 예측할 특징 차원 (y_dim_features)
                #  # ConvolutionalNetwork에 필요한 다른 파라미터들
                #  n_layers: int,
                #  n_channels: List[int],
                #  kernel_size_filter: List[int],
                #  kernel_size_pooling: List[int],
                #  stride_filter: List[int],
                #  stride_pooling: List[int]
                ):
        super().__init__()

        # first data: convolutional mode
        self.conv_net = ConvolutionalNetwork(
            c.conv_net_config, # dictionary
            # y_dim=y_dim_spectrum,
            # y_dim_features=spectrum_cnn_output_dim, # 이 CNN의 최종 출력 차원
            # n_layers=n_layers,
            # n_channels=n_channels,
            # kernel_size_filter=kernel_size_filter,
            # kernel_size_pooling=kernel_size_pooling,
            # stride_filter=stride_filter,
            # stride_pooling=stride_pooling
        )

        # 전역 파라미터 처리 모듈 초기화
        self.global_net = GlobalParamNetwork(
            c.global_net_config, # dictionary
            # input_dim=num_global_params,
            # output_dim=global_param_output_dim
        )

        # 통합된 특징을 최종 예측으로 변환하는 레이어
        combined_feature_dim = c.conv_net_config['out_dim_conv'] + c.global_net_config['out_dim_global']
        self.final_layer = nn.Linear(combined_feature_dim, c.y_dim_features)

    # def forward(self, spectrum_data: torch.Tensor, global_params: torch.Tensor) -> torch.Tensor:
    def forward(self, x: Union[Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor]]) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tuple[torch.Tensor, torch.Tensor] or List[torch.Tensor]
            A tuple/list containing two tensors:
            - x[0]: Data passing to conv_net (torch.Tensor, shape (Batch, channel, Data points))
                    (Channels first for Conv1d, e.g., (Batch, 2, 2000))
            - x[1]: Global parameters data passing to globa_linear (torch.Tensor, shape (Batch, Num_Global_Params))
                    (e.g., (Batch, 2) for (Rv_value, Rv_noise))

        Returns
        -------
        torch.Tensor
            Network output (features).
        """
        
        conv_data = x[0] # (Batch, channel, data points)
        global_data = x[1] # (Batch, data points)
        
        conv_features = self.conv_net(conv_data) # (Batch, conv_net_config['out_dim'])

        # 2. 전역 파라미터 처리
        global_features = self.global_net(global_data) # (Batch, glb_net_config['out_dim'])

        # 3. 특징 결합 (Concatenation)
        combined_features = torch.cat((conv_features, global_features), dim=1) # (Batch, combined_feature_dim)

        # 4. 최종 예측
        output = self.final_layer(combined_features)

        return output
    
    def features(self, x):
        return self.forward(x)

class ConvolutionalNetwork(nn.Module):
    """"
    1D Convolutional Feature net that accepts a configuration dictionary. 

    """
    def __init__(self, config_dict: Dict) -> None:
        super().__init__()

        # --- Extract necessary paramters ---
        # Extract and raise error if necessary parameters are missing
        try:
            self.in_dim = config_dict['in_dim_conv'] # number of convolution data points per channel
            self.out_dim = config_dict['out_dim_conv'] # final dimension of CNN
            # self.n_layers = config_dict['n_layers'] #  this is not given but calcualted within init
            self.start_channels = config_dict['start_channels'] # number of starting channel. n channel increase by x2 per layer
            self.in_channels = config_dict['in_channels'] # chanell of input data: noise-Net=2, normal-net=1
            self.kernel_size_filter = config_dict['kernel_size_filter'] # kernel size (one value)
            self.kernel_size_pooling = config_dict['kernel_size_pooling'] # one value
            self.stride_filter = config_dict['stride_filter'] # one value
            self.stride_pooling = config_dict['stride_pooling'] # one value
        except KeyError as e:
            raise ValueError(f"Missing required parameter in param_dict: {e}. Please check your config dictionary.")

        # --- 선택적 파라미터 추출 (기본값 설정) ---
        self.load_weights_path = config_dict.get('load_weights_path', None)

        # --- Build network
        layers = []
        # 입력 채널 수는 network config, prenoise_training에 따라 자동적으로 결정될 것.
        # Normal-Net:1, Noise-Net: 2 (플럭스 + 노이즈)로 설정
        # 만약 config에서 이 값을 제어하고 싶다면 config_dict.get('in_channels', 2) 등으로 변경 가능
        in_chan = self.in_channels
        out_chan = self.start_channels
        self.ch_size_final = self.in_dim # 초기 길이 (스펙트럼 길이)
        
        # 각 컨볼루션 레이어 블록 구성
        n=0
        while True:
            temp_len_after_conv = int(np.floor((self.ch_size_final - self.kernel_size_filter) / self.stride_filter + 1))
            # Calculate length after pooling for the potential next layer
            temp_len_after_pooling = int(np.floor((temp_len_after_conv - self.kernel_size_pooling) / self.stride_pooling + 1))
            temp_size = temp_len_after_pooling * out_chan
            # --- Stopping condition ---
            # Stop if the next potential length is less than or equal to the target out_dim
            # but current_len is still greater than out_dim.
            # Or if temp_len_after_pooling becomes <= 0 (too small to continue meaningful convolution)
            if temp_size <= self.out_dim or temp_len_after_pooling <= 0:
                # Ensure we have processed at least one layer to avoid empty network if in_dim is already small
                if n > 0 or self.in_dim <= self.out_dim: # If no layers added yet, but current_len is already small enough
                    break # Stop adding layers
                    
            # Another condition: for too many layers
            if n > 7 and temp_size <= self.out_dim * 2:
                break
                
            
            # --- Add current layer block if conditions are met ---
            # Convolution layer
            layers.append(nn.Conv1d(in_channels=in_chan,
                                    out_channels=out_chan,
                                    kernel_size=self.kernel_size_filter,
                                    stride=self.stride_filter))
            # Batch Normalisation
            layers.append(nn.BatchNorm1d(out_chan))
            
            # ReLu (Activation Function)
            layers.append(nn.ReLU())

            # Max pooling
            layers.append(nn.MaxPool1d(kernel_size=self.kernel_size_pooling,
                                       stride=self.stride_pooling))
            
            # Update channel trackers for the next layer
            in_chan = out_chan
            # out_chan = self.n_channels[n+1]
            out_chan = out_chan * 2
            self.ch_size_final = temp_len_after_pooling # Update current length to the new calculated length
            n += 1 # Increment layer counter
            # print(n, self.ch_size_final)
        self.final_channel = in_chan
        self.n_layers = n
        self.conv = nn.Sequential(*layers)

        # 최종 완전 연결 레이어 (이 CNN 모듈의 최종 출력 특징 차원 y_dim_features로 변환)
        # ch_size_final이 0보다 작거나 같아지는 경우를 방지하기 위한 안전장치
        if self.ch_size_final <= 0:
            raise ValueError(f"Convolutional layers reduce dimension too much. "
                             f"Final length is {self.ch_size_final}. Adjust kernel_size/stride or n_layers.")
        
        self.linear_out = nn.Linear(in_features=self.ch_size_final * self.final_channel,
                                    out_features=self.out_dim)

        # 미리 학습된 가중치 로드
        if self.load_weights_path is not None:
            state_dict = torch.load(self.load_weights_path)
            self.load_state_dict(state_dict)
            print(f"Loaded pretrained weights from {self.load_weights_path}")
            
            
    # def __init__(self, config_dict: Dict) -> None:
    #     super().__init__()

    #     # --- Extract necessary paramters ---
    #     # Extract and raise error if necessary parameters are missing
    #     try:
    #         self.in_dim = config_dict['in_dim'] # number of convolution data points per channel
    #         self.out_dim = config_dict['out_dim'] # final dimension of CNN
    #         self.n_layers = config_dict['n_layers'] # # of CNN layers
    #         self.n_channels = config_dict['n_channels'] # list of output channels for each layer. NOT the input channel
    #         self.in_channels = config_dict['in_channels'] # list of output channels for each layer. NOT the input channel
    #         self.kernel_size_filter = config_dict['kernel_size_filter'] # list of kernel size for each layer
    #         self.kernel_size_pooling = config_dict['kernel_size_pooling'] # list of  Maxpooling kernel size for each layer
    #         self.stride_filter = config_dict['stride_filter'] # list of stride size for each layer
    #         self.stride_pooling = config_dict['stride_pooling'] # list of pooling stride size for each layer
    #     except KeyError as e:
    #         raise ValueError(f"Missing required parameter in param_dict: {e}. Please check your config dictionary.")

    #     # --- 선택적 파라미터 추출 (기본값 설정) ---
    #     self.load_weights_path = config_dict.get('load_weights_path', None)

    #     # --- Build network
    #     layers = []
    #     # 입력 채널 수는 network config, prenoise_training에 따라 자동적으로 결정될 것.
    #     # Normal-Net:1, Noise-Net: 2 (플럭스 + 노이즈)로 설정
    #     # 만약 config에서 이 값을 제어하고 싶다면 config_dict.get('in_channels', 2) 등으로 변경 가능
    #     in_chan = self.in_channels
    #     out_chan = self.n_channels[0]
    #     self.ch_size_final = self.in_dim # 초기 길이 (스펙트럼 길이)

    #     # 각 컨볼루션 레이어 블록 구성
        
    #     for n in range(self.n_layers):
    #         # Calculate output size after conv and pooling
    #         # 패딩(padding)과 팽창(dilation)이 0과 1로 가정됨.
    #         # config_dict에 padding과 dilation을 포함하고 싶다면, 이 계산식도 수정해야 함.
    #         self.ch_size_final = int(np.floor((self.ch_size_final - self.kernel_size_filter[n]) / self.stride_filter[n] + 1))
    #         self.ch_size_final = int(np.floor((self.ch_size_final - self.kernel_size_pooling[n]) / self.stride_pooling[n] + 1))

    #         # Convolution layer
    #         layers.append(nn.Conv1d(in_channels=in_chan,
    #                                 out_channels=out_chan,
    #                                 kernel_size=self.kernel_size_filter[n],
    #                                 stride=self.stride_filter[n]))

    #         # Batch Normalisation
    #         layers.append(nn.BatchNorm1d(out_chan))

    #         # ReLu (Activation Function)
    #         layers.append(nn.ReLU())

    #         # Max pooling
    #         layers.append(nn.MaxPool1d(kernel_size=self.kernel_size_pooling[n],
    #                                    stride=self.stride_pooling[n]))

    #         # Update channel trackers for the next layer
    #         if n < self.n_layers - 1:
    #             in_chan = out_chan
    #             out_chan = self.n_channels[n+1]

    #     self.conv = nn.Sequential(*layers)

    #     # 최종 완전 연결 레이어 (이 CNN 모듈의 최종 출력 특징 차원 y_dim_features로 변환)
    #     # ch_size_final이 0보다 작거나 같아지는 경우를 방지하기 위한 안전장치
    #     if self.ch_size_final <= 0:
    #         raise ValueError(f"Convolutional layers reduce dimension too much. "
    #                          f"Final length is {self.ch_size_final}. Adjust kernel_size/stride or n_layers.")
        
    #     self.linear_out = nn.Linear(in_features=self.ch_size_final * out_chan,
    #                                 out_features=self.out_dim)

    #     # 미리 학습된 가중치 로드
    #     if self.load_weights_path is not None:
    #         state_dict = torch.load(self.load_weights_path)
    #         self.load_state_dict(state_dict)
    #         print(f"Loaded pretrained weights from {self.load_weights_path}")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Standard pytorch forward method.
        Parameters
        ----------
        x : torch.Tensor
            Input Tensor for the 1D CNN. Shape should be (Batch, Channels,  input_dim).
            Example: (Batch, 2, 2000) for flux and noise.

        Returns
        -------
        torch.Tensor
            Network output features from this CNN module. Shape (Batch, out_dim).
        '''
        y = self.conv(x)
        y = self.linear_out(torch.flatten(y, 1)) # Flatten from second dimension (dim=1)
        return y
    
    
    
class GlobalParamNetwork(nn.Module):
    def __init__(self, config_dict: Dict) -> None:
        super().__init__()
        # --- Extract necessary paramters ---
        # Extract and raise error if necessary parameters are missing
        try:
            self.in_dim = config_dict['in_dim_global'] # 
            self.out_dim = config_dict['out_dim_global'] # final dimension of gloab net
            self.n_layers = config_dict['n_layers_global'] # # of linear layers
        except KeyError as e:
            raise ValueError(f"Missing required parameter in param_dict: {e}. Please check your config dictionary.")
    # def __init__(self, input_dim: int, output_dim: int, num_layers: int = 2):
        
        layers = []
        in_dim = self.in_dim
        for i in range(self.n_layers):
            # 중간 레이어 차원 확장, 마지막은 고정
            out_dim = self.out_dim if i == self.n_layers - 1 else max(self.in_dim * 2, self.out_dim) 
            layers.append(nn.Linear(in_dim, out_dim))
            if i < self.n_layers - 1:
                layers.append(nn.ReLU())
            in_dim = out_dim
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    
    
    
######
# HybridStackedFeatureNet
######

class HybridStackedFeatureNet(nn.Module):
    def __init__(self, c):
        super().__init__()
        
        # # ==== Network building ====
        ## Networks: global_net, cnn_blocks, final_layer
        cnn_config_dict = c.conv_net_config
        gln_config_dict = c.global_net_config
        
        # --- Extract necessary paramters ---
        # Extract and raise error if necessary parameters are missing
        try:
            self.cnn_in_dim = cnn_config_dict['in_dim_conv'] # number of convolution data points per channel
            # self.cnn_out_dim = cnn_config_dict['out_dim_conv'] # final dimension of CNN => automatic. not set
            # self.n_layers = config_dict['n_layers'] #  this is not given but calcualted within init
            self.cnn_start_channels = cnn_config_dict['start_channels'] # number of starting channel. n channel increase by x2 per layer
            self.cnn_in_channels = cnn_config_dict['in_channels'] # Noise-Net:2, Normal-Net:1
            self.cnn_kernel_size_filter = cnn_config_dict['kernel_size_filter'] # kernel size (one value)
            self.cnn_kernel_size_pooling = cnn_config_dict['kernel_size_pooling'] # one value
            self.cnn_stride_filter = cnn_config_dict['stride_filter'] # one value
            self.cnn_stride_pooling = cnn_config_dict['stride_pooling'] # one value
            
            self.gln_in_dim = gln_config_dict['in_dim_global'] # 
            self.gln_out_dim = gln_config_dict['out_dim_global'] # final dimension of gloab net
            self.gln_n_layers = gln_config_dict['n_layers_global'] # # of linear layers
        except KeyError as e:
            raise ValueError(f"Missing required parameter in param_dict: {e}. Please check your config dictionary.")

        
        # 1) Make global network (linear)
        layers = []
        in_dim = self.gln_in_dim
        for i in range(self.gln_n_layers):
            # 중간 레이어 차원 확장, 마지막은 고정
            out_dim = self.gln_out_dim if i == self.gln_n_layers - 1 else max(self.gln_in_dim * 2, self.gln_out_dim) 
            layers.append(nn.Linear(in_dim, out_dim))
            if i < self.gln_n_layers - 1:
                layers.append(nn.ReLU())
            in_dim = out_dim
        self.global_net = nn.Sequential(*layers)
        
        
        # 2) Make CNN. n_layer 
        # 입력 채널 수는 network config, prenoise_training에 따라 자동적으로 결정될 것.
        # 이 경우 out_dim_conv도 정해놓지 않음.
        # Normal-Net:1, Noise-Net: 2 (플럭스 + 노이즈)로 설정
        # 만약 config에서 이 값을 제어하고 싶다면 config_dict.get('in_channels', 2) 등으로 변경 가능
        in_chan = self.cnn_in_channels
        out_chan = self.cnn_start_channels
        ch_size_final = self.cnn_in_dim # 초기 길이 (스펙트럼 길이)
        
        # prepare input dimension of the final linear network: just one layer
        self.final_in_dim = self.cnn_in_channels * self.cnn_in_dim  + self.gln_out_dim # before CNN
        
        
        n=0
        self.cnn_blocks = nn.ModuleList()
        while True:
            temp_len_after_conv = int(np.floor((ch_size_final - self.cnn_kernel_size_filter) / self.cnn_stride_filter + 1))
            # Calculate length after pooling for the potential next layer
            temp_len_after_pooling = int(np.floor((temp_len_after_conv - self.cnn_kernel_size_pooling) / self.cnn_stride_pooling + 1))
            temp_size = temp_len_after_pooling #* out_chan
            # --- Stopping condition ---
            # Stop if the next potential length is less than or equal to 2*(filter_kernel + filter_stride)
            if temp_size <= 2*(self.cnn_kernel_size_filter + self.cnn_stride_filter ):
                # Ensure we have processed at least one layer to avoid empty network if in_dim is already small
                if n > 0 or self.cnn_in_dim <= 2*(self.cnn_kernel_size_filter + self.cnn_stride_filter ): # If no layers added yet, but current_len is already small enough
                    break # Stop adding layers
            # Another condition: for too many layers
            if n > 7:
                break
                
            # --- Add current layer block if conditions are met ---
            layers = [] # set of conv layer 
            # Convolution layer
            layers.append(nn.Conv1d(in_channels=in_chan,
                                    out_channels=out_chan,
                                    kernel_size=self.cnn_kernel_size_filter,
                                    stride=self.cnn_stride_filter))
            # Batch Normalisation
            layers.append(nn.BatchNorm1d(out_chan))
            
            # ReLu (Activation Function)
            layers.append(nn.ReLU())

            # Max pooling
            layers.append(nn.MaxPool1d(kernel_size=self.cnn_kernel_size_pooling,
                                       stride=self.cnn_stride_pooling))
            
            
            # Add dimension to final network: flattend output + global net output
            self.final_in_dim += (temp_len_after_pooling * out_chan + self.gln_out_dim )
            
            # Update channel trackers for the next layer
            in_chan = out_chan
            # out_chan = self.n_channels[n+1]
            out_chan = out_chan * 2
            ch_size_final = temp_len_after_pooling # Update current length to the new calculated length
            n += 1 # Increment layer counter
            # print(n, self.ch_size_final)
            
            
            
            self.cnn_blocks.append( nn.Sequential(*layers) ) # each layer(block) will be used separately
            
        self.cnn_final_channel = in_chan # number of channels in the final CNN layer
        self.cnn_n_layers = n 
        
        # 3) Final linear network (1 layer): output dim: c.y_dim_features
        self.final_layer = nn.Linear(self.final_in_dim, c.y_dim_features)
        
        
    # Input format is the same as HybridFeatureNet: tuple of spectral and glbal data
    def forward(self, x: Union[Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor]]) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tuple[torch.Tensor, torch.Tensor] or List[torch.Tensor]
            A tuple/list containing two tensors:
            - x[0]: Data passing to conv_net (torch.Tensor, shape (Batch, channel, Data points))
                    (Channels first for Conv1d, e.g., (Batch, 2, 2000))
            - x[1]: Global parameters data passing to globa_linear (torch.Tensor, shape (Batch, Num_Global_Params))
                    (e.g., (Batch, 2) for (Rv_value, Rv_noise))
                    
        Returns
        -------
        torch.Tensor
            Network output (features).
        """
        
        current_conv_output = x[0] # (Batch, channel, data points)
        global_data = x[1] # (Batch, data points)
        
        # 1) Get global features
        global_features = self.global_net(global_data) # (Batch, glb_net_config['out_dim'])
        
        # 2) Pass CNN and get output of each layer. append glabal feature. make final
        # first: before passing CNN
        combined_list = [ torch.cat( (torch.flatten(current_conv_output, 1), global_features), 1) ]
        
        for i, conv_block in enumerate(self.cnn_blocks):
            current_conv_output = conv_block(current_conv_output)
            combined_list.append( torch.cat( (torch.flatten(current_conv_output, 1), global_features), 1) )
        
        combined_features = torch.cat(combined_list, 1)
        
        # 3) Pass final layer
        output = self.final_layer(combined_features)
        
        return output
    
    def features(self, x):
        return self.forward(x)
        

    










