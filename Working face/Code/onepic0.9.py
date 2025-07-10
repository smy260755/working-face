import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# 设置数据文件的路径
data_folder = '../data/'
files = [f'639_{i:02d}th.xlsx' for i in range(10, 15)]  # 从639_03th到639_08th

# 创建颜色列表，使用更明显的颜色
colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']

# 创建一个画布
plt.figure(figsize=(12, 8))

for file, color in zip(files, colors):
    try:
        # 读取数据文件
        data = pd.read_excel(os.path.join(data_folder, file))

        xs = data['Out1'].values.reshape(-1, 1)  # 不再进行标准化
        ys = data['Out2'].values.reshape(-1, 1)  # 不再进行标准化

        # 使用 DBSCAN 聚类算法
        dbscan = DBSCAN(eps=0.3, min_samples=100)
        dbscan.fit(np.column_stack((xs, ys)))

        # 获取聚类标签
        labels = dbscan.labels_

        # 找出频率最高的轨迹对应的索引
        cluster_indices = [i for i, l in enumerate(labels) if l != -1]  # 排除噪声点

        # 获取频率最高的聚类标签对应的轨迹数据
        xs_cluster = xs[cluster_indices]
        ys_cluster = ys[cluster_indices]

        # 将数据合并为二维数组，以便进行 PCA
        data_pca = np.column_stack((xs_cluster, ys_cluster))

        # 应用 PCA
        pca = PCA(n_components=2)
        pca.fit(data_pca)

        # 计算原始数据点在第一和第二主成分上的投影
        projected_data = data_pca @ pca.components_.T

        # 计算包含 90%数据点的边界
        quantile_90 = np.percentile(projected_data, 90, axis=0)
        quantile_10 = np.percentile(projected_data, 10, axis=0)

        # 计算偏移量
        offset_90 = (quantile_90 - quantile_10) / 2

        # 计算矩形的四个顶点
        center = np.mean(data_pca, axis=0)
        rectangle_vertices = np.array([
            center + pca.components_[0] * offset_90[0] + pca.components_[1] * offset_90[1],
            center + pca.components_[0] * offset_90[0] - pca.components_[1] * offset_90[1],
            center - pca.components_[0] * offset_90[0] - pca.components_[1] * offset_90[1],
            center - pca.components_[0] * offset_90[0] + pca.components_[1] * offset_90[1]
        ])

        # 画出矩形
        plt.plot(
            [rectangle_vertices[0, 0], rectangle_vertices[1, 0]],
            [rectangle_vertices[0, 1], rectangle_vertices[1, 1]],
            color=color,
            label=file[:-5]  # 使用文件名（去掉扩展名）作为标签
        )
        plt.plot(
            [rectangle_vertices[1, 0], rectangle_vertices[2, 0]],
            [rectangle_vertices[1, 1], rectangle_vertices[2, 1]],
            color=color
        )
        plt.plot(
            [rectangle_vertices[2, 0], rectangle_vertices[3, 0]],
            [rectangle_vertices[2, 1], rectangle_vertices[3, 1]],
            color=color
        )
        plt.plot(
            [rectangle_vertices[3, 0], rectangle_vertices[0, 0]],
            [rectangle_vertices[3, 1], rectangle_vertices[0, 1]],
            color=color
        )
    except Exception as e:
        print(f"处理文件 {file} 时发生错误: {e}")

# 设置坐标轴单位长度相等
plt.axis('equal')

# 添加图例
plt.legend(title="id_data", bbox_to_anchor=(0.85, 1), loc='upper left')

# 设置标题
plt.title('working face')

# 显示图形
plt.tight_layout()
plt.show()