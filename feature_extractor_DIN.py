# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:42:04 2023

@author: Administrator
"""

import torch
# import torch.nn.functional as F
# from torch.utils import data
# from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
#%%
class deeplob_DIN(nn.Module):
    def __init__(self, y_len):
        super().__init__()
        self.y_len = y_len
        #convolution part
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=(3, 1), stride=(3, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(3)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=(5, 1), stride=(5, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(3)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1), padding='same'),
            # nn.BatchNorm2d(4)
            )
        
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x_list = [x1, x2, x3]
        x_result = torch.cat(x_list, dim=1)
        
        return x_result

#%%
''' Original Custom Inception Module '''

class OrigCIM_DIN(nn.Module):
    def __init__(self, t_filter, n_asset):
        super().__init__()
        self.t_filter = t_filter
        self.n_asset = n_asset
        self.orig_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 1), stride=(1, 1)),
            nn.ELU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=(self.t_filter, 1), stride=(self.t_filter, 1)),
            nn.ELU(),
            nn.BatchNorm2d(3)
            )
        
        self.orig_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 1), stride=(1, 1)),
            nn.ELU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=(1, self.n_asset), stride=(1, self.n_asset)),
            nn.ELU(),
            nn.BatchNorm2d(3)
            )
        
        self.orig_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 1), stride=(1, 1)),
            nn.ELU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=(self.t_filter, self.n_asset), stride=(self.t_filter, self.n_asset)),
            nn.ELU(),
            nn.BatchNorm2d(3)
            )
        
        self,orig_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(1, 1), stride=(1, 1)),
            nn.ELU(),
            nn.BatchNorm2d(4)
            )
        
    def forward(self, x, t_filter, n_asset):
        x1 = self.orig_layer1(x, t_filter, n_asset)
        x2 = self.orig_layer2(x, t_filter, n_asset)
        x3 = self.orig_layer3(x, t_filter, n_asset)
        x4 = self.orig_layer4(x, t_filter, n_asset)
        
        x_list = [x1, x2, x3, x4]
        x_result = torch.cat(x_list, dim=1)
        return x_result
    
#%%
'''Flexible Custom Inception Module'''
class FlexCIM(nn.Module):
    def __init__(self, input_layer1, input_layer2, input_layer3, output_layer_1x1, filter_num1, filter_num2, filter_num3):
        super().__init__()
        self.input_layer1 = input_layer1
        self.input_layer2 = input_layer2
        self.input_layer3 = input_layer3
        self.output_layer_1x1 = output_layer_1x1
        self.layer_1x1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.output_layer_1x1, kernel_size=(1, 1), padding='same'),
            nn.ELU(),
            nn.BatchNorm2d(self.output_layer_1x1)
            )
        self.layer_5x1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.input_layer1, kernel_size=(5, 1), padding='same'),
            nn.ELU(),
            nn.BatchNorm2d(self.input_layer1)
            )
        self.layer_5x1_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_layer1, out_channels=self.input_layer1, kernel_size=(5, 1), padding='same'),
            nn.ELU(),
            nn.BatchNorm2d(self.input_layer1)
            )
        self.layer_1x10 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.input_layer2, kernel_size=(1, 10), padding='same'),
            nn.ELU(),
            nn.BatchNorm2d(self.input_layer2)
            )
        self.layer_1x10_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_layer2, out_channels=self.input_layer2, kernel_size=(1, 10), padding='same'),
            nn.ELU(),
            nn.BatchNorm2d(self.input_layer2)
            )
        self.layer_5x10 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.input_layer3, kernel_size=(5, 10), padding='same'),
            nn.ELU(),
            nn.BatchNorm2d(self.input_layer3)
            )
        self.layer_5x10_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_layer3, out_channels=self.input_layer3, kernel_size=(5, 10), padding='same'),
            nn.ELU(),
            nn.BatchNorm2d(self.input_layer3)
            )
        self.filter_num1 = filter_num1
        self.filter_num2 = filter_num2
        self.filter_num3 = filter_num3
        
    def forward(self, x):
        X1 = x.clone()
        X2 = x.clone()
        X3 = x.clone()
        layer_result = []
        result1 = self.layer_1x1(x)
        layer_result.append(result1)
        for i in range(self.filter_num1):
            if (i == 0):
                X1 = self.layer_5x1(X1)
                layer_result.append(X1)
            else:
                X1 = self.layer_5x1_2(X1)
                layer_result.append(X1)
        # print(i)
        for j in range(self.filter_num2):
            if (j == 0):
                X2 = self.layer_1x10(X2)
                layer_result.append(X2)
            else:
                X2 = self.layer_1x10_2(X2)
                layer_result.append(X2)
        # print(j)
        for t in range(self.filter_num3):
            if (t == 0):
                X3 = self.layer_5x10(X3)
                layer_result.append(X3)
            else:
                X3 = self.layer_5x10_2(X3)
                layer_result.append(X3)
        # print(t)
        X_result = torch.cat(layer_result, dim=1)
        return X_result
#%%
'''Dimensionality reduction module'''
class dimension_reduction(nn.Module):
    def __init__(self, input_dim, hidden_layer_rdt):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layer = hidden_layer_rdt
        self.conv_rdt1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim, out_channels=self.hidden_layer, kernel_size=(1, 1), stride=(1, 1)),
            nn.ELU()
            )
        self.conv_rdt2 = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_layer, out_channels=1, kernel_size=(1, 1), stride=(1, 1)),
            nn.ELU()
            )
        
    def forward(self, X):
        X = self.conv_rdt1(X)
        X = self.conv_rdt2(X)
        return X
#%%
