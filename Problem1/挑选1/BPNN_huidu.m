% 2021-07-02 ANN人工神经网络-assignment2
% Author: Chenfeng Yuan
% E-mail: chenfengyuan.hb@gmail.com
% 数据来源论文链接 dataset CLAY_6_535_TC304.xlsx

% 导出参数
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


tic
clear;
close all;
clc % clear the command window

rand('seed', 1);
% randn('seed', 1);

hiddenminno=10; % BPNN的隐含层数量
best=5; % 
% cd 'D:\01-learning\05-CQU\人工神经网络\作业2 Assignment 2' % 更改当前工作目录到包含数据文件的路径
% data=load('BPF.txt'); % MCS训练数据535*8，前7列是对应指标，第8列是预测强度数据
data1=xlsread('关联度data.xlsx');  % MCS训练数据535*8，前7列是对应指标，第8列是预测强度数据
% 训练数据
train_ratio = 0.8; % 训练集占总数据集的比例
data_train = data1(1:1974*train_ratio,:);
% 测试数据
% data2=load('BPFtesting.txt');
data_test = data1(1974*train_ratio:1974,:);

% train_ratio = 0.5; % 训练集占总数据集的比例
%  = data(1:535*train_ratio,:);% 训练数据
%  = data(535*train_ratio:535,:);% 测试数据
 for run=1:1:best
    
     maxepoch =500;

% trainset;训练集
temp_train=data_train';
traininput1=temp_train(1:20,:);
traintarget1=temp_train(21,:);
% normalize
[traininput,mintraininput1,maxtraininput1,traintarget,mintraintarget1,maxtraintarget1] = premnmx(traininput1, traintarget1);
% temp_test=(load('tunnel testing.txt'))';
temp_test=data_test';
testinput1=temp_test(1:20,:);
testtarget=temp_test(21,:); 
 [testinput] = tramnmx(testinput1,mintraininput1,maxtraininput1);
 [PN,minp,maxp,TN,mint,maxt] = premnmx(testinput1,testtarget);

[attrno,trainexpno] = size(traininput);% 'attrno' is the number of attributes, 'trainexpno' is the number of training examples

% rand('state',sum(100*clock)); 

    
    % train the  neural networks
    net = newff(minmax(traininput),[hiddenminno 1],{'logsig' 'purelin'});
    net.trainParam.epochs = maxepoch;
    net.trainParam.goal = 0.0;
    net.trainParam.max_fail=100;
    net.trainParam.min_grad=1e-15;
    net.trainParam.mu_max=1e20;
    net.trainParam.mu_dec=0.7;
    net.trainParam.mu_inc=1.03;
    net.trainParam.lr=0.01;
net = train(net,traininput,traintarget);
 

% test 


[n,testexpno] = size(testinput);                              % 'testexpno' is the number of test examples, 'n' is useless
 [m,trexpno] = size(traininput);     
 output2 = zeros(1,trexpno);
 output1 = zeros(1,testexpno);  
      output1 = sim(net,testinput);   
            [output_te] = postmnmx(output1,mintraintarget1,maxtraintarget1);
          output2 = sim(net,traininput);   
            [output_tr] = postmnmx(output2,mintraintarget1,maxtraintarget1);  
            
                       
            
    figure(run);
    [m,b,r]=postreg(output_te,testtarget);
      
   mse_te = mse(output_te - testtarget);                          % obtain the mean squared error of the ensemble    
   mse_tr = mse(output_tr - traintarget1);    
   mse_te1 = mse(output1 - TN)  ;                        % according to the scaled value   
   mse_tr2 = mse(output2 - traintarget)   ;     
   fprintf('single  hidden=  %g  run= %g     r= %-12.5g   mse_te=  %-12.5g    mse_tr=  %-12.5g',hiddenminno, run,r, mse_te, mse_tr);
   fprintf('\n\n');
 end
 toc
% end of function


   