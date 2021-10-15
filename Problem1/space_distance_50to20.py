#%%
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

# path = r'D:\cf-projects\01-科研\2021研究生数学建模\02-working\MathModel2021\Problem1\关联度data.xlsx'
path = r'D:\cf-projects\01-科研\2021研究生数学建模\02-working\MathModel2021\Problem1\关联度_50.xlsx'
# raw_data = pd.read_excel(path) # 原始数据，包含表头
# data = raw_data.values # 去除表头的数据
data1 = pd.read_excel(path)

# data = 
#%%
factors = data1.values # 其它因素
# s = data[:, 0] # 硫含量
# RON = data[:, 1] # 辛烷值
# factors = data[:, 2:] # 其它因素

m, n = factors.shape
corrs = np.zeros((n, n))
for i in range(n):
    print(i)
    for j in range(i, n):
        corrs[i, j] = distcorr(factors[:, i], factors[:, j])

print(corrs)
frame = pd.DataFrame(corrs)
frame.to_excel(r'D:\cf-projects\01-科研\2021研究生数学建模\02-working\MathModel2021\Problem1\corr_50_to_20.xlsx', "Sheet2")

#%% 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_ads = pd.read_excel('./corr_50_to_20.xlsx',header=None)


plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.figure(figsize=(10, 10))
plt.rcParams['font.sans-serif'] = ['Times New Roman']
# sns.heatmap(df_ads.corr(), cmap="Reds", annot = True,annot_kws={"size":8})
# sns.heatmap(df_ads.corr(method='spearman'), cmap="Reds", annot = True, annot_kws={"size":5})
sns.heatmap(df_ads.iloc[1:,1:], cmap="Reds", annot = True, annot_kws={"size":5})
plt.tight_layout()
plt.title("Spearman rank Correlation Coefficient MAttrix Heatmap")
plt.savefig("./随机森林+灰度预测50to20-热力图.png",dpi=1000)
plt.show()
# %%
