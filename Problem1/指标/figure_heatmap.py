#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_ads = pd.read_excel('./挑选1.xlsx',header=None)

df_ads1 = df_ads.iloc[2:,:20]

df_1 = df_ads1.apply(lambda x:x.astype(float))

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.figure(figsize=(10, 10))
plt.rcParams['font.sans-serif'] = ['Times New Roman']
# sns.heatmap(df_ads.corr(), cmap="Reds", annot = True,annot_kws={"size":8})
# sns.heatmap(df_ads.corr(method='spearman'), cmap="Reds", annot = True, annot_kws={"size":5})
sns.heatmap(df_1.corr(), cmap="Reds", annot = True, annot_kws={"size":5})
plt.tight_layout()
plt.title("Spearman rank Correlation Coefficient MAttrix Heatmap")
plt.savefig("./挑选1-热力图.png",dpi=600)
plt.show()
# %%
