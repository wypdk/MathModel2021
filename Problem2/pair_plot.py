

#%% 挑选1
import pandas as pd

import seaborn as sns

import matplotlib as plt

data = pd.read_excel('挑选1.xlsx',header=None) # 1975行，21列
data1 =  data.iloc[1:,:21]
# 
sns.pairplot(data1,size=2.5)
            
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("指标",fontsize=14) # 添加x轴的名称
plt.ylabel("指标",fontsize=14) # 添加y轴的名称
plt.tight_layout()
plt.savefig("./散点图.png",dpi=1000)
plt.show()

#%% 挑选2

import pandas as pd

import seaborn as sns

import matplotlib as plt

path = "D:\\cf-projects\\01-科研\\2021研究生数学建模\\02-working\\MathModel2021\\Problem1\\挑选2\\挑选2.xlsx"

data = pd.read_excel(path, sheet_name='2', header=None) # 1975行，21列
data1 =  data.iloc[1:,:21]
# 
sns.pairplot(data1,size=2.5)
# %% 挑选3
import pandas as pd

import seaborn as sns

import matplotlib as plt

path = "D:\\cf-projects\\01-科研\\2021研究生数学建模\\02-working\\MathModel2021\\Problem1\\挑选2\\挑选2.xlsx"

data = pd.read_excel(path, sheet_name='3', header=None) # 1975行，21列
data1 =  data.iloc[1:,:21]
# 
sns.pairplot(data1,size=2.5)