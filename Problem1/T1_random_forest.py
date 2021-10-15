#%% 
# 通过随机森林分离变量对辛烷值含量的重要性
from time import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

path = r'D:\cf-projects\01-科研\2021研究生数学建模\02-working\MathModel2021\Problem1\data_extract.xlsx'

path_alpha = r'D:\cf-projects\01-科研\2021研究生数学建模\02-working\MathModel2021\Problem1\ER_activity.xlsx'
data = pd.read_excel(path) # 原始数据，包含表头
# data1 = data.values # 去除表头数据

# ER = pd.read_excel("ER_activity.xlsx", sheet_name="training")
# ER = ER.values

ER = pd.read_excel("ER_activity.xlsx")
# ER = pd.read_excel("path_alpha")
ER.head()
#%% 
ER_alpha = ER.iloc[:,2]

#%%

# s = data[0]
# RON = data[1] # 辛烷值

#%%
factors = data.iloc[:,2:] # 其它因素

# a.drop(0,axis = 1,inplace = True)  # 删除列，返回的新数组被替换，保存在a中
# data.drop(data.columns([0:2],axis = 1))   #连续
#%%
factors = factors.dropna(axis=1)

t0 = time()

forest = RandomForestRegressor(n_estimators=500,
random_state=0,
max_features=100,
n_jobs=2)

X_train, X_test, y_train, y_test = train_test_split(factors, ER_alpha,
test_size=0.3,shuffle=True,random_state=0)

forest.fit(X_train, y_train)

print("done in %0.3fs" % (time() - t0))

importances = forest.feature_importances_
print(importances)

print(sorted(importances, reverse=True))

#%%

array1 = sorted(importances, reverse=True)

# percent = array1.sum(:20)
percent = 0
for i in range(100):
    percent += array1[i]

print(percent)
print(array1[100])
#%%
index = []
index = [col for col in factors]


df = pd.DataFrame(importances)

df1 = df.T

#%%
df1.columns = index
#%%
df2 = df1.sort_values(by=0, axis=1, ascending=False)
#%% 根据列名合并2个dataframe

# data_100 = pd.merge(df2, factors.astype(float))

df2.to_excel("data_top100.xlsx")

#%% 生成指标序列前100列

# for x in 
                      