# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 13:53:44 2023

@author: Administrator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

def data_processor(price, bench, start_int, end_int):
    '''
    用于将数据均切片到目标时间范围

    Parameters
    ----------
    price : dataframe
        price data.
    bench : dataframe
        benchmark data.
    start_int : int
        start date.
    end_int : TYPE
        end date.

    Returns
    -------
    price1 : dataframe
        processed price data.
    bench1 : dataframe
        processed benchamrk data.

    '''

    price1 = price.set_index('date')
    price1 = price1.dropna(axis=1)
    bench1 = bench[bench['trade_date']>start_int]
    bench1 = bench1[bench1['trade_date']<=end_int]
    bench1.set_index('trade_date', inplace=True)
    bench1 = np.log(bench1) - np.log(bench1.shift(1))
    bench1.fillna(0, inplace=True)
    return price1, bench1


# def train_test_split(price, bench, percentage=0.75):
#     train_per = int(price.shape[0] * percentage)
#     price_train = torch.tensor(price.iloc[:train_per, :].values, dtype=torch.float32)
#     price_test = torch.tensor(price.iloc[train_per:, :].values, dtype=torch.float32)
#     bench_train = torch.tensor(bench.iloc[:train_per, :].values, dtype=torch.float32)
#     bench_test = torch.tensor(bench.iloc[train_per:, :].values, dtype=torch.float32)
#     return price_train, price_test, bench_train, bench_test


def ewm(data, span):
    '''
    计算指数移动平滑

    Parameters
    ----------
    data : tensor
        origin data.
    span : int
        .

    Returns
    -------
    weighted_avg : tensor
        data with ewm.

    '''
    
    alpha = 2 / (span + 1)

    weighted_avg = torch.zeros_like(data)
    weighted_avg[0] = data[0]

    for i in range(1, len(data)):
        weighted_avg[i] = alpha * data[i] + (1 - alpha) * weighted_avg[i - 1]

    return weighted_avg


# def ewm_std(data, span):
#     ewm_mean = ewm(data, span)
#     ewm_square_mean = ewm(data ** 2, span)
#     ewm_std = torch.sqrt(ewm_square_mean - ewm_mean ** 2)
#     return ewm_std


'''2024/1/16 改的'''
def rolling_std(tensor, window_size):
    '''
    计算滚动波动率

    Parameters
    ----------
    tensor : tensor
        raw data(return data).
    window_size : int
        window size.

    Returns
    -------
    std : tensor
        rolling standard deviation.

    '''
    unfolded = tensor.unfold(0, window_size, 1)
    mean = unfolded.mean(dim=2)
    squared_diff = (unfolded - mean.unsqueeze(0)).pow(2)
    variance = squared_diff.mean(dim=0)
    std = variance.sqrt()
    return std

def ewm_vol_new(price, span=63, window_size=21):
    '''
    计算滚动波动率后移动指数平滑的结果
    '''
    price_shift = torch.roll(price, shifts=1, dims=0)
    price_shift[0] = 1
    ret_data = torch.log(price) - torch.log(price_shift)
    vol_roll = ret_data.unfold(0, int(window_size), 1)
    vol = torch.std(vol_roll, dim=2)
    # ret_df = pd.DataFrame(ret_data.numpy())
    # vol = ret_df.rolling(window_size).std().iloc[window_size-1:, :]
    # vol = torch.tensor(vol.values, dtype=torch.float32)
    # print(vol)
    # vol = rolling_std(ret_df, window_size=window_size)
    ewm_v = ewm(vol, span=span)
    return ewm_v

'''之前写的错误的计算方式'''
# def ewm_vol(price, window_size=63):
#     price_shift = torch.roll(price, shifts=1, dims=0)
#     price_shift[0] = 1
#     ret_df = torch.log(price) - torch.log(price_shift)
#     vol = ewm_std(ret_df, span=window_size)
#     vol[torch.isinf(vol)] = 0
#     return vol
'''到这里'''

def scaled_volatility(price, span=63, window_size=21, sigma_scale=0.15):
    '''
    计算波动率调整（常数） / 滚动波动率（移动指数平滑后）
    '''
    vol = ewm_vol_new(price, span, window_size)
    vol_scale = sigma_scale / vol
    vol_scale[torch.isinf(vol_scale)] = 0
    return vol_scale


def volatility_adjusted_return(price, span=63, window_size=21):
    '''
    计算收益率 / 滚动波动率（移动指数平滑后）

    '''
    price_shift = torch.roll(price, shifts=1, dims=0)
    price_shift[0] = 1
    ret_df = torch.log(price) - torch.log(price_shift)
    '''这里的波动率是否要变成年化是个问题，我感觉可能不用'''
    vol = ewm_vol_new(price, span, window_size) * torch.sqrt(torch.tensor(252 / window_size))
    length = vol.shape[0]
    ret_df = ret_df[-length:]
    ret_ada = ret_df / vol
    ret_ada[torch.isnan(ret_ada)] = 0
    ret_ada[torch.isinf(ret_ada)] = 0
    return ret_ada


def return_shift(price, span=63, window_size=21, sigma_scale=0.15):
    '''
    计算volatility_adjusted_return下上挪一天（格）

    '''
    price_shift = torch.roll(price, shifts=1, dims=0)
    price_shift[0] = 1
    ret_df = torch.log(price) - torch.log(price_shift)
    vol = ewm_vol_new(price, span, window_size)
    length = vol.shape[0]
    ret_df = ret_df[-length:]
    ret_shift = torch.roll(ret_df, shifts=-1) / torch.roll(vol, shifts=-1)
    ret_shift[torch.isinf(ret_shift)] = 0
    ret_shift[torch.isnan(ret_shift)] = 0
    ret_shift *= sigma_scale
    return ret_shift


def vol_adjreturn(num_asset, price, weight_df, cost, span=63, window_size=21, sigma_scale=0.15):
    '''
    结合模型的输出weight，计算出整个投资组合的收益，考虑了换手的手续费，用两天的权重矩阵除以波动率，两天做差乘交易费率cost
    '''
    Y_t1 = return_shift(price, span, window_size, sigma_scale)
    # price_shift = torch.roll(price, shifts=1, dims=0)
    # price_shift[0] = 1
    # ret_df = torch.log(price) - torch.log(price_shift)
    # Y_t1 = torch.roll(ret_df, shifts=-1)
    Y_t1 = Y_t1.to(torch.float32)
    Y_t2 = scaled_volatility(price, span, window_size, sigma_scale)
    vol = ewm_vol_new(price, span, window_size)
    # print(Y_t1)
    vol[0] = 1
    w_vol = weight_df / vol
    # w_vol = weight_df
    w_vol[torch.isinf(w_vol)] = 0
    w_vol[torch.isnan(w_vol)] = 0
    # print(w_vol)
    ret = torch.diagonal(torch.matmul(weight_df, Y_t1.t()) / num_asset)
    # print('最小：',torch.min((1 + ret).cumprod(dim=0)).float())
    # print('最大：', torch.max((1 + ret).cumprod(dim=0)).float())
    # print(max_drawdown_torch((1 + ret).cumprod(dim=0)))
    cost_1 = (cost * sigma_scale / num_asset) * torch.sum(torch.diff(w_vol, dim=0, prepend=w_vol[:1]), dim=1)
    # cost_1 = (cost * sigma_scale / num_asset) * torch.sum(w_vol, dim=1)
    adj_ret = ret - cost_1
    # adj_ret = ret
    return adj_ret

def weight_loss(tensor):
    ind = torch.max(torch.abs(tensor)).item()
    if (ind > 1):
        loss = ind
    else:
        loss = 0
    return loss

def max_drawdown_torch(vec):
    '''
    

    Parameters
    ----------
    vec : tensor
        return tensor.

    Returns
    -------
    tensor
        max drawdowns(tensor).

    '''
    maximums, _ = torch.cummax(vec, dim=0)
    drawdowns = 1 - vec / maximums
    return torch.max(drawdowns)


def position(tensor):
    tensor[tensor < 0] = -1
    tensor[tensor > 0] = 1
    return tensor