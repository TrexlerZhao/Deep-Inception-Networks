# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 20:54:43 2024

@author: Trexlers's elf
"""

import torch
import torch.nn as nn
import os
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
# 当前文件的路径
print('current file path'+os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))
from return_function_DINs import *
from train_test_function import *
from feature_extractor_DIN import FlexCIM, dimension_reduction
from position_sizer_DINs import PS_lstm
from loss_function_DINs import ReturnLoss, ReturnLoss_adj
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
#%%
# 网络模型
class Flex_Net(nn.Module):
    def __init__(self, config):
        super(Flex_Net, self).__init__()

        self.input_layer1 = config['input_layer1']
        self.input_layer2 = config['input_layer2']
        self.input_layer3 = config['input_layer3']
        self.output_layer_1x1 = config['output_layer_1x1']
        self.filter_num1 = config['filter_num1']
        self.filter_num2 = config['filter_num2']
        self.filter_num3 = config['filter_num3']
        self.num_assets = config['num_assets']
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.input_dim = config['input_dim']
        self.hidden_layer_rdt = config['hidden_layer_rdt']

        self.fe = FlexCIM(self.input_layer1, self.input_layer2, self.input_layer3, self.output_layer_1x1, self.filter_num1, self.filter_num2, self.filter_num3)
        self.dim_rdt = dimension_reduction(self.input_dim, self.hidden_layer_rdt)
        self.ps = PS_lstm(self.num_assets, self.input_size, self.hidden_size, self.num_layers)
                
    def forward(self, x):
        x = self.fe(x)
        x_rdt = self.dim_rdt(x)
        batch_size, channels, num_rows, num_columns = x_rdt.size()
        sequence_length = num_rows  # 将每一行作为一个时间步
        feature_dim = channels * num_columns  # 将通道数和列数相乘作为特征维度
        batch_size = 1
        reshaped_x = x_rdt.reshape(sequence_length, batch_size, feature_dim)
        # print(reshaped_x.shape)
        result = self.ps(reshaped_x)
        # result = torch.tanh(result)
        # result = position(result)
        # result = torch.sin(result)
        # result = torch.tanh(result * 10)
        # res = F.softmax(result, dim=1)
        return result
#%%
# 读取文件
price = pd.read_csv('price.csv')
bench = pd.read_csv('benchmark.csv')
# 处理数据
data = data_processor(price, bench, 20120101, 20231013)
price1 = data[0]
bench1 = data[1]
# 划分测试训练数据，并分割测试集
price_train = price1.iloc[:-768, :]
bench_train = bench1.iloc[:-768, :]
price_dict = extract_ranges(price1, 256, 768, 3)
bench_dict = extract_ranges(bench1, 256, 768, 3)
# 分割训练集数据
end = int(price_train.shape[0])
bench_list = []
train_list = []
for i in range(3):
    train_list.append(price1.iloc[i * 256:end + i * 256, :])
    bench_list.append(bench1.iloc[i * 256:end + i * 256, :])
# 设置GPU训练
device = torch.device("cuda:0")

#%%
# 定义模型参数
config = {
    'input_layer1': 3,
    'input_layer2': 3,
    'input_layer3': 3,
    'output_layer_1x1': 3,
    'filter_num1': 3,
    'filter_num2': 3,
    'filter_num3': 3,
    'num_assets': int(price_train.shape[1]),
    'input_size': int(price_train.shape[1]),
    'hidden_size': 128,
    'num_layers': 3,
    'input_dim': 30,
    'hidden_layer_rdt': 15
}
# 定义模型
network = Flex_Net(config)
network = network.to(device)
# 定义优化器
optimizer = optim.Adam(network.parameters(), lr=0.0005)
#%%
'''循环训练模型，但不能保证模型一定好，即这个模型的泛化能力和鲁棒性不强'''
ret_list = []
for i in range(0, 3):
    train = train_list[i]
    bench = bench_list[i]
    network_test = Flex_Net(config)
    network_test = network_test.to(device)
    optimizer = optim.Adam(network_test.parameters(), lr=0.0005)
    ret = train_test(train, price_dict[str(i)], bench, bench_dict[str(i)], 450, network_test, optimizer, device, 276, 'flexcim_test4'+str(i), 256, 276, 0.05, loss_type=0, overlap_size=22)
    ret_list.append(ret)
#%%
i = 0
train = train_list[i]
bench = bench_list[i]
network_test = Flex_Net(config)
network_test = network_test.to(device)
optimizer = optim.Adam(network_test.parameters(), lr=0.001)
ret = train_test(train, price_dict[str(i)], bench, bench_dict[str(i)], 450, network_test, optimizer, device, 276, 'flexcim_loss'+str(i), 256, 256, 3, loss_type=3)
#%%
train1 = torch.tensor(train.values, dtype=torch.float32)
z = volatility_adjusted_return(train1)
z1 = return_shift(train1)
print(z)
print(z1)
#%%
del ReturnLoss_plus