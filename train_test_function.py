# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:17:13 2024

@author: Trexlers's elf
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from return_function_DINs import *
from loss_function_DINs import ReturnLoss, ReturnLoss_adj, ReturnLoss_plus, ReturnLoss_scaled, ReturnLoss_new
    
def extract_ranges(df, m, n, num_parts):
    '''
    用于分割测试集的函数，分割测试集用来对不同时间段进行建模

    '''
    # df = pd.read_csv(csv_file_path)  # 读取CSV文件

    extracted_data = {}
    total_rows = df.shape[0]
    start_index = total_rows - n - m

    # 根据倒数行数和m的值计算范围
    for i in range(num_parts):
        start = start_index + (n // num_parts * i)
        end = start +m+  (n // num_parts)
        extracted_data[str(i)] = df.iloc[start:end, :]

    return extracted_data

def batch_list(data_list, batch, window):
    length = len(data_list)
    list0 = []
    # print(window)
    for i in range(int(int(data_list[0].shape[0] - batch) / window)):
        list1 = []
        for j in range(length):
            tensor = data_list[j]
            list1.append(tensor[window * i: batch + window * (i + 1)])
        list0.append(list1)
    list1 = []
    for j in range(length):
        list1.append(data_list[j][-batch:])
    list0.append(list1)
    return list0

# 定义画loss图函数
def update_loss_plot(loss_values):
    '''
    
    用于画loss图，观察是否合理收敛
    
    '''
    plt.figure(figsize=(12, 8), dpi=200)
    plt.plot(loss_values, 'b-')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('loss curve')
    plt.grid(True)
    plt.pause(0.05)  # 添加延迟以实现动态绘图

def model_train(price_train, bench_train, model, optimizer, num_epochs, device, batch_size, save_name, threshold, loss_type, span=63, window_size=21, overlap_size=22):
    '''
    
    用于模型训练的整个过程，会呈现每步的Loss变化（画图），同时最后输出模型，并且将训练好的模型按照输入的名字进行保存（原始路径）

    Parameters
    ----------
    price_train : tensor
        price data to train.
    bench_train : tensor
        benchmark data to train.
    model : model
        origin model.
    optimizer : optimizer
        optimizer like ADAM.
    num_epochs : int
        As the name suggests.
    device : device
        As the name suggests.
    batch_size : int
        As the name suggests.
    save_name : str
        name for the model parameter to save.
    threshold : float
        early stop threshold.
    loss_type : int
        now the value list includes 0,1,2,3,4.
    span : int, optional
        span parameter for the ewm. The default is 63.
    window_size : int, optional
        window size parameter for the moving volatility computation. The default is 21.

    Returns
    -------
    model : model
        the model after parameter tuning.

    '''
    price_train = price_train.to(device)
    x = volatility_adjusted_return(price_train)
    bench_train = bench_train.to(device)
    length = x.shape[0]
    price_train = price_train[-length:]
    bench_train = bench_train[-length:]
    data_list = [x, price_train, bench_train]
    loader = batch_list(data_list, batch_size, overlap_size)
    print(len(loader))
    # dataset = TensorDataset(x, price_train, bench_train)
    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss_values = []
    for epoch in range(num_epochs):
        for data in loader:
            # print(x[:20])
            x = data[0]
            p = data[1]
            bench = data[2]
            x = x[(window_size-1):]
            x1 = torch.unsqueeze(torch.unsqueeze(x, 0), 0).to(device)
            optimizer.zero_grad()  # 梯度清零
            outputs = model(x1)  # 前向传播
            outputs = outputs.squeeze(dim=1)
            if (loss_type == 0):
                criterion = ReturnLoss(2182, p, 0.002, bench, 0.01)
            elif (loss_type == 1):
                criterion = ReturnLoss_adj(2182, p, 0.002, bench, 0.01)
            elif (loss_type == 2):
                criterion = ReturnLoss_plus(2182, p, 0.002, bench, 0.01)
            elif (loss_type == 3):
                criterion = ReturnLoss_scaled(2182, p, 0.002, bench, 0.01)
            elif (loss_type == 4):
                criterion = ReturnLoss_new(2182, p, 0.002, bench, 0.01)
            loss = criterion(outputs)  # 计算损失
            loss.backward()  # 反向传播
        # max_norm = 1.0  # 设置梯度裁剪的阈值
        # torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm)
            optimizer.step()  # 更新参数
        loss_values.append(loss.cpu().detach().numpy())
        update_loss_plot(loss_values)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
        # 早停
        ptp = np.abs(np.ptp(np.array(loss_values)[-4:]))
        if ((epoch > 60) & (ptp < threshold)):
            print('This model should early stop')
            break
    torch.save(model.state_dict(), save_name+'.pth')
    return model


def model_backtest(model, test, device, test_len, fit_len):
    '''
    

    Parameters
    ----------
    model : model
        origin model.
    test : tensor
        test data.
    device : device
        As the name suggests.
    test_len : int
        every time the length of result that the model will predict.
    fit_len : int
        every time the length of result that the model use to predict.

    Returns
    -------
    weight_tensor : tensor
        the model output.

    '''
    weight_list = []
    with torch.no_grad():
        for i in range(test_len):
            data = test[i:(fit_len+i), :]
            data = torch.unsqueeze(torch.unsqueeze(data, 0), 0).to(device)
            res = model(data)
            # if (res[-1].sum() < 0):
            #     weight = -res[-1] / res[-1].sum()
            # else:
            #     weight = res[-1] / res[-1].sum()
            weight = res[-1]
            # weight[weight < 0] = -1
            # weight[weight > 0] = 1
            # weight[weight < 0] = -weight[weight < 0] / weight[weight < 0].sum()
            # weight[weight > 0] = weight[weight > 0] * 2 / weight[weight > 0].sum()
            # weight = res[-1] / res[-1].sum()
            # weight = res[-1] / torch.abs(res[-1]).sum()
            weight_list.append(weight)
            del data
            torch.cuda.empty_cache()
    weight_tensor = torch.cat(weight_list, dim=0)
    return weight_tensor

def pnl_calculate(ret, weight, index, bench, save_name):
    '''
    输入原始收益率，权重矩阵和时间index，来计算投资组合收益及画图

    Parameters
    ----------
    ret : tensor
        return origin data.
    weight : tensor
        weight data, the model output.
    index : dataframe
        index list(time index).
    bench : dataframe
        benchmark data.

    Returns
    -------
    ret_prod : datafarame
        final return data.

    '''
    # weight = weight / 1000
    ret_p = torch.diag(torch.matmul(weight, ret.t()))
    ret = ret_p.cpu().numpy()
    ret_prod = pd.DataFrame((1 + ret).cumprod())
    ret_prod.index = pd.to_datetime(index)
    bench.index = pd.to_datetime(index)
    bench = (1 + bench).cumprod()
    fig, ax = plt.subplots(figsize=(15, 8), dpi=200)
    # ax.set_xticklabels(pd.to_datetime(index)[:7], rotation=45)
    plt.xticks(rotation=45)
    ax.plot(ret_prod, label='Portfolio')
    ax.plot(bench, label='Benchmark')
    ax.set_title('Return of Portfolio')
    ax.set_xlabel('Time')
    ax.set_ylabel('Return')
    ax.legend()
    plt.savefig(save_name + '.png')
    plt.show()
    return ret_prod, ax

def evaluation_indicator(res, benchmark):
    '''
    输入投资组合收益率序列和benchmark收益率序列，来求各项指标。

    Parameters
    ----------
    res : dataframe
        portfolio return data.
    benchmark : dataframe
        benchmark return data.

    Returns
    -------
    eval_df : dataframe
        .

    '''
    # 指标计算
    eval_df = pd.DataFrame(index=['累计收益率', '年化收益率', '最大回撤', '夏普比率', '年化超额收益率',
                           '月最大超额收益', '跑赢基准日占比', '正收益月份占比'], columns=res.columns)
    for i in range(int(res.shape[1])):
        days = len(res.index)
        ret = res.iloc[:, i]
        ret.index = benchmark.index
        # 计算有效年数
        yearnum = len(res.index) / 250
        # 累计收益率
        return_cump = np.around(np.cumprod(ret + 1).iloc[-1], 4)
        eval_df.iloc[0, i] = str(np.around(return_cump-1, 4)*100) + '%'
        # 年化收益率
        annul = (return_cump) ** (1 / yearnum) - 1
        eval_df.iloc[1, i] = str(np.round(annul * 100, 2)) + '%'
        # 最大回撤
        cummax = (res + 1).cumprod().iloc[:, i].cummax()
        maxback = ((cummax - (res + 1).cumprod().iloc[:, i]) / cummax).max()
        eval_df.iloc[2, i] = str(np.around(maxback*100, 2)) + '%'
        # 夏普比率
        eval_df.iloc[3, i] = np.around((annul - 0.04) / (ret.std() * np.sqrt(250)), 2)
        # 年化超额收益率
        alpha = (ret - pd.Series(benchmark.iloc[:, 0]) + 1).cumprod().iloc[-1]
        alpha_ann = (alpha) ** (1 / yearnum) - 1
        eval_df.iloc[4, i] = str(np.round((alpha_ann) * 100, 2)) + '%'
        # 月最大超额收益
        excess_return = ret - pd.Series(benchmark.iloc[:, 0])
        excess_return1 = excess_return.astype('float')
        eval_df.iloc[5, i] = str(np.round((excess_return1).max() * 100, 2)) + '%'
        # 跑赢基准概率
        eval_df.iloc[6, i] = str(np.round((excess_return1 > 0).sum() / days * 100, 2)) + '%'
        # 正收益日占比
        eval_df.iloc[7, i] = str(np.round((ret > 0).sum() / days * 100, 2)) + '%'
    
    return eval_df

def train_test(train_price, test_price, train_bench, test_bench, num_epochs, ori_model, optimizer, device, batch_size, save_name, test_len, fit_len, threshold, loss_type=0, overlap_size=22):
    '''
    
    将model_train函数、train_test函数、pnl_calculate函数结合，打包成一个整体训练加测试函数（比较方便，但debug不方便）

    '''
    train = torch.tensor(train_price.values, dtype=torch.float32)
    train_bench = torch.tensor(train_bench.values, dtype=torch.float32)
    model = model_train(train, train_bench, ori_model, optimizer, num_epochs, device, batch_size, save_name, threshold, loss_type, overlap_size)
    test_price1 = torch.cat([train[-20:], (torch.tensor(test_price.values))], dim=0)
    x = volatility_adjusted_return(torch.tensor(test_price1, dtype=torch.float32))
    ret_test = test_price.pct_change(1).shift(-1).fillna(0)
    ret_test = ret_test.iloc[255:]
    index = ret_test.index[1:]
    weight = model_backtest(model, x, device, test_len, fit_len).to(device)
    ret = torch.tensor(ret_test.values, dtype=torch.float32).to(device)
    ret_p, ax = pnl_calculate(ret, weight, index, test_bench[256:], save_name)
    return ret_p