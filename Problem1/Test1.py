# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 19:47:13 2021

@author: chenfengyuan

@E-mail: chenfengyuan@cqu.edu.cn
    
@Site: Chongqing University 

@Goal:
"""


#%% test

import pandas as pd

path = r'D:\cf-projects\01-科研\2021研究生数学建模\02-working\MathModel2021\Problem1\Molecular_Descriptor.xlsx'
raw_data = pd.read_excel(path) # 原始数据，包含表头

data = raw_data.values # 去除表头的数据

# 计算每个指标的均值

data_t = raw_data

col_mean = raw_data.mean()
# 剔除掉均值小于0.1的指标, 
#%% 
index = []
# for i in range(0,729):
for i in range(0,729):   
    # print(i)
    if col_mean[i] < 0.1:
        index.append(i)

data_t.drop(columns = data_t.columns[index],
        inplace = True)
#%% 

data_t.to_excel(".\data_extract.xlsx")

# index = [col for col in raw_data]

# for index
# df = raw_data
# data_new = raw_data.drop(df[(df.B>9) & (df.B < 12)].index)

#%% 


#%% 变量之间的距离相关系数计算
# 相关性分析

from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd

def distcorr(X, Y):
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))

    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    
    return dcor

path = r'D:\cf-projects\01-科研\2021研究生数学建模\02-working\MathModel2021\Molecular_Descriptor.xlsx'
raw_data = pd.read_excel(path) # 原始数据，包含表头
data = raw_data.values # 去除表头的数据

factors = data[:, 1:] # 其它因素
# s = data[:, 0] # 硫含量
# RON = data[:, 1] # 辛烷值
# factors = data[:, 2:] # 其它因素

m, n = factors.shape

#%%
corrs = np.zeros((n, n))
for i in range(n):
    print(i)
    for j in range(i, n):
        corrs[i, j] = distcorr(factors[:, i], factors[:, j])

print(corrs)
frame = pd.DataFrame(corrs)
frame.to_excel(r'D:\cf-projects\01-科研\2021研究生数学建模\02-working\MathModel2021\corr.xlsx', "Sheet2")

#%% 

