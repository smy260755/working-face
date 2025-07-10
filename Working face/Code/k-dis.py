import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# 读取Excel文件
data = pd.read_excel('../data/604_15th.xlsx')

# 提取Out1和Out2列作为特征
X = data[['Out1', 'Out2']].values

# 选择K值，通常选择K=2*数据维度-1
k = 4

# 计算K-distance
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)
k_distances = distances[:, k-1]

# 对K-distance值进行排序
k_distances = np.sort(k_distances)

# 绘制K-distance图
plt.plot(k_distances)
plt.title("K-dist Plot")
plt.xlabel("Points")
plt.ylabel("K-distance")
plt.show()