# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 09:06:01 2023

@author: Administrator
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
#%%
class PS_lstm(nn.Module):
    def __init__(self, num_assets, input_size, hidden_size, num_layers):
        super().__init__()
        self.n_assets = num_assets
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_layer = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.n_assets)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm_layer(x, (h0, c0))
        out = self.fc(out)  # 取最后一个时间步的输出
        # out = out / out.sum(dim=1, keepdim=True)
        # print('lstm done')
        return out
    
#%%
        