# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 22:15:50 2021

@author: chenfengyuan

@E-mail: chenfengyuan@cqu.edu.cn
    
@Site: Chongqing University 

@Goal:
"""

#%% 随机森林方法
import numpy as np
import pandas as pd

from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet

from sklearn.svm import SVR

# import sklearn.ensemble.GradientBoostingRegressor # 集成算法
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score #交叉验证

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score

import joblib

import matplotlib.pyplot as plt

import seaborn as sns

#%matplotlib inline
#%%
#1.数据导入
data = pd.read_excel('D:\\cf-projects\\01-科研\\2021研究生数学建模\\02-working\\MathModel2021\\Problem2\\挑选1.xlsx',header=None) # 1975行，21列
data
#%% 
# train_data =pd.read_excel('RFR_train.xlsx',index_col=0)
# train_data_1 =pd.read_excel('RFR_train.xlsx',index_col=0)
# y_train = train_data.pop("P_RON")

train_data = data.iloc[1:,:20]

train_data1 = data.iloc[1:,:20]

train_data
y_train = data.iloc[1:,20]
y_train
#%% 设置交叉验证次数
n_folds = 5

# # 普通线性回归
# # 通过加入一个参数 kind='reg' seaborn 可以添加一条最佳拟合直线和 95% 的置信带。
# sns.pairplot(train_data1, x_vars=['RON','SZORB.TC_2801.PV','S ZORB.PDT_1004.DACA'],
# y_vars='P_RON', size=7, aspect=0.8,kind = 'reg')
# # plt.savefig("pairplot.jpg")
# plt.show()

#%% 
lr_model = LinearRegression()
joblib.dump(lr_model,'saved_lr_model')
lr_model.fit(train_data,y_train)
a = lr_model.intercept_ # 截距
b = lr_model.coef_ # 回归系数
#%% 
# 梯度增强回归模型对象
gbr_model = GradientBoostingRegressor()
joblib.dump(gbr_model,'saved_gbr_model')

# 随机森林回归模型
rfr_model = RandomForestRegressor()
joblib.dump(rfr_model,'saved_rfr_model')
# 不同模型的名称列表

model_names = ['LinearRegression', 'GBR','RandomForestRegressor']
# 不同回归模型
# model_dic = [gr_model, lr_model, etc_model, svr_model, gbr_model,rfr_model]
model_dic = [ lr_model, gbr_model,rfr_model]
# 交叉验证结果

cv_score_list = []
#%%

# 各个回归模型预测的 y 值列表
pre_y_list = []
# 读出每个回归模型对象
for model in model_dic:
    # 将每个回归模型导入交叉检验
    scores = cross_val_score(model, train_data, y_train, cv=n_folds)
    # 将交叉检验结果存入结果列表
    cv_score_list.append(scores)
    # 将回 归训练中得到的预测 y 存入列表
    pre_y_list.append(model.fit(train_data, y_train).predict(train_data))
#%%
###模型效果指标评估
# 获取样本量，特征数
n_sample, n_feature = train_data.shape
# 回归评估指标对象列表
model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]
# 回归评估指标列表
model_metrics_list = []
# 循环每个模型的预测结果
for pre_y in pre_y_list:
    # 临时结果列表
    tmp_list = []
    # 循环每个指标对象
    for mdl in model_metrics_name:
        # 计算每个回归指标结果
        tmp_score = mdl(y_train, pre_y)
        # 将结果存入临时列表
        tmp_list.append(tmp_score)
    # 将结果存入回归评估列表
    model_metrics_list.append(tmp_list)
df_score = pd.DataFrame(cv_score_list, index=model_names)# 各个交叉验证的结果
df_met = pd.DataFrame(model_metrics_list, index=model_names, columns=['ev', 'mae', 'mse','r2'])

#%%# 各 种评估结果
### 可视化
# 创建画布
plt.figure(figsize=(16, 36))
# 颜色列表
color_list = ['r', 'g', 'b']
# 循环结果画图
for i, pre_y in enumerate(pre_y_list):

    # 子网络
    plt.subplot(3, 1, i+1)
    # 画出原始值的曲线
    plt.plot(np.arange(train_data.shape[0]), y_train, color='k', label='y')
    # 画出各个模型的预测线
    plt.plot(np.arange(train_data.shape[0]), pre_y, color_list[i], label=model_names[i])
    plt.title(model_names[i])
    plt.legend(loc='lower left')
plt.savefig('T2回归v1.png', dpi=600)
plt.show()


#%% XGBoost方法


#%% Adaboost


#%% keras DNN


#%% LSTM


#%% 多元
