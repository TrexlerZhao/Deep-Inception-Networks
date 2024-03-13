# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 14:51:50 2023

@author: Administrator
"""

import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
# 当前文件的路径
print('current file path'+os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))
from return_function_DINs import *

class ReturnLoss(nn.Module):
    '''
    最普通的Loss，年化夏普比率加上与基准的相关性
    '''
    def __init__(self, num_asset, price, cost, benchmark, corr_cost, span=63, window_size=21, sigma_scale=0.15):
        super(ReturnLoss, self).__init__()
        self.num_asset = num_asset
        self.price = price
        self.cost = cost
        self.benchmark = benchmark
        self.window_size = window_size
        self.sigma_scale = sigma_scale
        self.corr_cost = corr_cost
        self.span = span

    def forward(self, weight_df):
        # weight_df[weight_df < 0] = -1
        # weight_df[weight_df > 0] = 1
        # weight_df = weight_df / 100
        # weight_df = torch.sign(weight_df) / 100
        return_df = vol_adjreturn(self.num_asset, self.price, weight_df, self.cost, self.span, self.window_size, self.sigma_scale)
        # res = torch.sum(return_df)
        # print(torch.max(1 + return_df1).cumprod(dim=0))
        # print(torch.min(1 + return_df1).cumprod(dim=0))
        # print(return_df)
        return_df1 = return_df.unsqueeze(0).t()
        mat = torch.cat([return_df1, self.benchmark[(self.window_size-1):]], dim=1).t()
        # print(mat)
        corr = torch.abs(torch.corrcoef(mat)[0, 1])
        # print(corr)
        # loss1 = -np.sqrt(252) * torch.mean(return_df) / torch.std(return_df)
        loss1 = -torch.mean(return_df) / torch.std(return_df)
        # print(loss1)
        loss2 = self.corr_cost * corr
        # loss3 = torch.abs(weight_df.sum(dim=1, keepdim=True) - 1).sum()
        # loss = loss1 + loss2 + loss3
        loss = loss1 + loss2
        return loss.float()
    

class ReturnLoss_scaled(nn.Module):
    '''
    调整后的Loss，主要惩罚每一天权重总和加起来远小于1的部分（因为这个会导致标准化后权重巨大，会导致收益率跳）
    其他的公式写在报告里了。
    '''
    def __init__(self, num_asset, price, cost, benchmark, corr_cost, span=63, window_size=21, sigma_scale=0.15):
        super(ReturnLoss_scaled, self).__init__()
        self.num_asset = num_asset
        self.price = price
        self.cost = cost
        self.benchmark = benchmark
        self.window_size = window_size
        self.sigma_scale = sigma_scale
        self.corr_cost = corr_cost
        self.span = span

    def forward(self, weight_df):
        return_df = vol_adjreturn(self.num_asset, self.price, weight_df, self.cost, self.span, self.window_size, self.sigma_scale)
        # res = torch.sum(return_df)
        return_df1 = return_df.unsqueeze(0).t()

        mat = torch.cat([return_df1, self.benchmark[(self.window_size-1):]], dim=1).t()
        # print(mat)
        corr = torch.abs(torch.corrcoef(mat)[0, 1])
        # print(corr)
        loss1 = -np.sqrt(252) * torch.mean(return_df) / torch.std(return_df)
        # print(loss1)
        loss2 = self.corr_cost * corr
        w = weight_df.sum(dim=1, keepdim=True)
        alpha = 1 / 5
        w[(torch.abs(w) - 1) < -0.5] = alpha * torch.log(1 / torch.abs(w[(torch.abs(w) - 1) < -0.5]))
        thres = 0.5
        w[(torch.abs(w) > 1) & ((torch.abs(w) - 1) < thres)] = 0
        w[torch.abs(w) > 1.5] = torch.abs(w[torch.abs(w) > 1.5]) - 1
        loss3 = w.sum()
        # print(loss3)
        loss = loss1 * 3 + loss2 + loss3
        # loss = loss1 + loss2
        return loss.float()


class ReturnLoss_adj(nn.Module):
    '''
    另外一种惩罚权重总和的方式。但效果不好
    '''
    def __init__(self, num_asset, price, cost, benchmark, corr_cost, span=63, window_size=21, sigma_scale=0.15):
        super(ReturnLoss_adj, self).__init__()
        self.num_asset = num_asset
        self.price = price
        self.cost = cost
        self.benchmark = benchmark
        self.window_size = window_size
        self.sigma_scale = sigma_scale
        self.corr_cost = corr_cost
        self.span = span


    def forward(self, weight_df):
        return_df = vol_adjreturn(self.num_asset, self.price, weight_df, self.cost, self.span, self.window_size, self.sigma_scale)
        # res = torch.sum(return_df)
        return_df1 = return_df.unsqueeze(0).t()
        # print(torch.max(1 + return_df).cumprod(dim=0))
        # print(torch.min(1 + return_df).cumprod(dim=0))
        # print(return_df)
        
        mat = torch.cat([return_df1, self.benchmark[(self.window_size-1):]], dim=1).t()
        # print(mat)
        corr = torch.abs(torch.corrcoef(mat)[0, 1])
        # print(corr)
        loss1 = -np.sqrt(252) * torch.mean(return_df) / torch.std(return_df)
        # print(loss1)
        loss2 = self.corr_cost * corr
        loss3 = torch.abs(weight_df.sum(dim=1, keepdim=True) - 1).sum() / 1.25
        print(loss3)
        loss = loss1 + loss2 + loss3
        # loss = loss1 + loss2
        return loss.float()
    
    
class ReturnLoss_plus(nn.Module):
    '''
    进阶的损失函数构造，多加了一项是关于总收益、正收益比率和最大回撤的，但在实际的测试中这个损失函数有问题。
    '''
    def __init__(self, num_asset, price, cost, benchmark, corr_cost, span=63, window_size=21, sigma_scale=0.15):
        super(ReturnLoss_plus, self).__init__()
        self.num_asset = num_asset
        self.price = price
        self.cost = cost
        self.benchmark = benchmark
        self.window_size = window_size
        self.sigma_scale = sigma_scale
        self.corr_cost = corr_cost
        self.span = span

    def forward(self, weight_df):
        return_df = vol_adjreturn(self.num_asset, self.price, weight_df, self.cost, self.span, self.window_size, self.sigma_scale)
        # res = torch.sum(return_df)
        return_df1 = return_df.unsqueeze(0).t()
        # print(return_df)
        
        mat = torch.cat([return_df1, self.benchmark[(self.window_size-1):]], dim=1).t()
        # print(mat)
        corr = torch.abs(torch.corrcoef(mat)[0, 1])
        # print(corr)
        loss1 = -np.sqrt(252) * torch.mean(return_df) / torch.std(return_df)
        # print(torch.mean(return_df))
        # print(torch.std(return_df))
        # print(loss1)
        loss2 = self.corr_cost * corr
        loss3 = torch.abs(weight_df.sum(dim=1, keepdim=True) - 1).sum() / 3
        zero = torch.sum(torch.gt(return_df1, 0)) / return_df1.shape[0]
        # print(zero)
        dd = max_drawdown_torch((1 + return_df1).cumprod(dim=0))
        # print(torch.min((1 + return_df1).cumprod(dim=0)))
        # print(torch.max(1 + return_df1))
        # print(torch.min(1 + return_df1))
        ret = (1 + return_df1).cumprod(dim=0)[-1] - 1
        # print(ret)
        loss4 = -(((zero / 100) + ret) / dd) / 10
        # print(loss4)
        loss = loss1 + loss2 + loss3 + loss4
        # loss = loss1 + loss2
        return loss.float()
    
    
    
class ReturnLoss_new(nn.Module):
    '''
    另外一种惩罚权重总和的方式。但效果不好
    '''
    def __init__(self, num_asset, price, cost, benchmark, corr_cost, span=63, window_size=21, sigma_scale=0.15):
        super(ReturnLoss_new, self).__init__()
        self.num_asset = num_asset
        self.price = price
        self.cost = cost
        self.benchmark = benchmark
        self.window_size = window_size
        self.sigma_scale = sigma_scale
        self.corr_cost = corr_cost
        self.span = span


    def forward(self, weight_df, loss_mul=1):
        return_df = vol_adjreturn(self.num_asset, self.price, weight_df, self.cost, self.span, self.window_size, self.sigma_scale)
        # res = torch.sum(return_df)
        return_df1 = return_df.unsqueeze(0).t()
        
        mat = torch.cat([return_df1, self.benchmark[(self.window_size-1):]], dim=1).t()
        # print(mat)
        corr = torch.abs(torch.corrcoef(mat)[0, 1])
        # print(corr)
        loss1 = -torch.mean(return_df) / torch.std(return_df)
        # print(loss1)
        loss2 = self.corr_cost * corr
        loss3 = loss_mul * weight_loss(weight_df)
        # print(loss3)
        loss = loss1 + loss2 + loss3
        # loss = loss1 + loss2
        return loss.float()