import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# 读取 new.xlsx 文件
data = pd.read_excel('../tezheng.xlsx')

xs = data['Out1']  # 将 Out1 数据转换为非负值
ys = data['Out2']  # 将 Out2 数据转换为非负值

# 对 x 和 y 进行标准化
scaler = StandardScaler()
xs_scaled = scaler.fit_transform(xs.values.reshape(-1, 1))
ys_scaled = scaler.fit_transform(ys.values.reshape(-1, 1))

# 使用 DBSCAN 聚类算法
dbscan = DBSCAN(eps=0.3, min_samples=5)  # 参数可根据实际情况进行调整
dbscan.fit(xs_scaled)

# 获取聚类标签
labels = dbscan.labels_.tolist()  # 将标签转换为列表

# 创建一个全零的数组来存储计数
counts = np.zeros(max(labels) + 1)

# 统计每个标签的出现次数
for label in labels:
    counts[label] += 1

# 获取出现频率最高的标签索引
most_common_cluster = np.argmax(counts)
print(most_common_cluster)

# 找出频率最高的轨迹对应的索引
cluster_indices = [i for i, l in enumerate(labels) if l == most_common_cluster]

# 获取频率最高的聚类标签对应的轨迹数据
xs_cluster = np.array(xs[cluster_indices])
ys_cluster = np.array(ys[cluster_indices])

# 进行线性回归拟合
model = LinearRegression()
model.fit(xs_cluster.reshape(-1, 1), ys_cluster)

# 计算曲线拟合的决定系数
r2 = r2_score(ys_cluster, model.predict(xs_cluster.reshape(-1, 1)))
print("曲线拟合的决定系数:", r2)

# 计算平均绝对误差
mae = mean_absolute_error(ys_cluster, model.predict(xs_cluster.reshape(-1, 1)))
print("平均绝对误差:", mae)

# 计算边界线的偏移量
offset = np.percentile(ys_cluster - model.predict(xs_cluster.reshape(-1, 1)), 5)

# 为不同标签的点设置不同颜色
cluster_colors = {i: np.random.rand(3,) for i in set(labels)}

# 绘制原始数据和拟合曲线以及边界线（在拟合曲线上方）
for label, color in cluster_colors.items():
    cluster_indices = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(xs[cluster_indices], ys[cluster_indices],label=f'Cluster {label}')

#plt.plot(xs_cluster, model.predict(xs_cluster.reshape(-1, 1)) - offset, color='green', linestyle='--', label='Boundary curve')
plt.legend()
plt.show()