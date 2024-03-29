% 2021-07-02 ANN人工神经网络-assignment2
% Author: Chenfeng Yuan
% E-mail: chenfengyuan.hb@gmail.com
% 数据来源论文链接 dataset CLAY_6_535_TC304.xlsx

% 不排水剪切强度预测
% 桩的一些参数，
% 使用7个指标预测一个y：Su/sigma v
% Depth (m)	深度
% σ'v (kPa)	竖向有效应力
% su/σ'v	不排水剪切强度/竖向有效应力
% OCR	over consolidation ratio 超固结比
% (qt-σv)/σ'v	（矫正锥尖电阻-总有效应力）/竖向有效应力
% (qt-u2)/σ'v	（矫正锥尖电阻-锥体后面的压力）/竖向有效应力
% (u2-u0)/σ'v 	(锥体后面的压力-静水孔隙压力) /竖向有效应力
% Bq 孔隙率
% 数据 535*8

clc
clear
close all

% cd 'D:\cf-projects\01-科研\2021研究生数学建模\02-working\MathModel2021\Problem1'

tic

% cd 'D:\01-learning\05-CQU\人工神经网络\作业2 Assignment 2' % 更改当前工作目录到包含数据文件的路径
data=xlsread('关联度data.xlsx'); % MCS训练数据535*8，前7列是对应指标，第8列是预测强度数据
% 训练数据

%%
train_ratio = 0.7; % 训练集占总数据集的比例
% a = data1(:,:);
% 测试数据
% data2=load('BPFtesting.txt');

% b = data2(:,:);
a = data(1:1974*train_ratio,:);
b = data(1974*train_ratio:1974,:);


%% 

% X1=a(1:535*train_ratio,:);
% X2=a(:,2);
% X3=a(:,3);
% X4=a(:,4);
% X5=a(:,5);
% X6=a(:,6);
% X7=a(:,7);


% X=[X1,X2,X3,X4,X5,X6,X7];
X = a(:,1:20);

Y= a(:,21);
params = aresparams(40, [], false, [], [], 2); % MARS模型对应的参数
% 46 代表最大基函数数量 Number of Basis Function
% 基函数包括一次项、二次项；
% 17个指标，那么基函数数量最大值可以是17*17=289,排列组合
% 综合考虑：准确性，鲁棒性，参数调优

% 参数调优实例：
% 基函数数量 r^2(training) R^2(testing)
% 42 0.9444 0.9516
% 44 0.9464 0.9523
% 46 0.9478 0.9564
% 48 0.9466 0.9490
% 50 0.9503 0.9555
% 100 0.9645 0.9670

model = aresbuild(X, Y, params) 
aresanova(model,X,Y) % MARS方差分析
Yq = arespredict(model, X);
mu=mean(Y);
J=sum((Yq-Y).^2); % 真实值-预测值
S=sum((Y-mu).^2); % 真实值-均值
tmse=sum((Y-Yq).^2);
mse=tmse/length(Y);
r2=1-J/S % 反映训练数据的一个准确性&精确度
% b=load('MCStesting.txt'); % MCS测试数据994*18


% Xt=[Xt1,Xt2,Xt3,Xt4,Xt5,Xt6,Xt7,Xt8,Xt9,Xt10,Xt11,Xt12,Xt13,Xt14,Xt15,Xt16,Xt17];
Xt = b(:,1:20);
Yt=b(:,21);

% [MSE, RMSE, RRMSE, R2] = arestest(model, Xt, Yt)
R2 = arestest(model, Xt, Yt)
results= arestest(model, Xt, Yt)
aresplot(model)
areseq(model,5)
Yq_test = arespredict(model, Xt);

toc

% 展示模型的基函数 areseq(model,3)


