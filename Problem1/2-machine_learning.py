##### data introduction
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as  plt

root_path = os.getcwd()
wd = np.array(['0', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'])  # 如果是数值型就不可以，坐标轴的恶补不连续性呢
################0    1     2    3   4     5     6     7     8     9    10    11     12   13    14    15
load_data = np.loadtxt("./data_feature.txt")
load_label = np.loadtxt("./std_response.txt").reshape(-1, 1)
##### plot picture
plt.plot(wd, load_label, ls='-', lw=2, label='cosine', color='purple')
plt.show()
# -------------------
# %%
from pycaret.regression import *

# X_train, X_test, y_train, y_test = train_test_split(load_data, load_label, test_size=0.2, random_state=42,shuffle=False)
d_k =  1
d_w =  13
g_k =  13
g_w = 16
X_train = load_data[d_k:d_w, :]
X_test = load_data[g_k:g_w, :]
y_train = load_label[d_k:d_w, :]
y_test = load_label[g_k:g_w, :]
# %%
trained_data = np.concatenate((y_train, X_train), axis=1)
test_data = np.concatenate((y_test, X_test), axis=1)
trained_data = pd.DataFrame(trained_data)
test_data = pd.DataFrame(test_data)

str_values = []
str_values.append("label")
clo_values = np.arange(1, 37, 1)  # 37 , 6
for i in clo_values:
    str_values.append(str(i))
test_data.columns = str_values
trained_data.columns = str_values
# %%
reg1 = setup(fold=5, preprocess=False, data=trained_data, train_size=1.0, target='label', test_data=test_data)
compare_models()
# %%
# aaa = reg1[22][1]
# aaa.to_csv("./data2.csv")
