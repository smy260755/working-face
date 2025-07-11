import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os


data_folder = '../data/'
files = [f'639_{i:02d}th.xlsx' for i in range(10, 15)]  


colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']

plt.figure(figsize=(12, 8))

for file, color in zip(files, colors):
    try:
  
        data = pd.read_excel(os.path.join(data_folder, file))

        xs = data['Out1'].values.reshape(-1, 1) 
        ys = data['Out2'].values.reshape(-1, 1)  

    
        dbscan = DBSCAN(eps=0.3, min_samples=100)
        dbscan.fit(np.column_stack((xs, ys)))

  
        labels = dbscan.labels_

        # Find the index corresponding to the trajectory with the highest frequency
        cluster_indices = [i for i, l in enumerate(labels) if l != -1]  

        xs_cluster = xs[cluster_indices]
        ys_cluster = ys[cluster_indices]

        data_pca = np.column_stack((xs_cluster, ys_cluster))

        #Apply PCA
        pca = PCA(n_components=2)
        pca.fit(data_pca)

        # Calculate the projection of raw data points onto the first and second principal components
        projected_data = data_pca @ pca.components_.T

        quantile_90 = np.percentile(projected_data, 90, axis=0)
        quantile_10 = np.percentile(projected_data, 10, axis=0)

        offset_90 = (quantile_90 - quantile_10) / 2

  
        center = np.mean(data_pca, axis=0)
        rectangle_vertices = np.array([
            center + pca.components_[0] * offset_90[0] + pca.components_[1] * offset_90[1],
            center + pca.components_[0] * offset_90[0] - pca.components_[1] * offset_90[1],
            center - pca.components_[0] * offset_90[0] - pca.components_[1] * offset_90[1],
            center - pca.components_[0] * offset_90[0] + pca.components_[1] * offset_90[1]
        ])

 
        plt.plot(
            [rectangle_vertices[0, 0], rectangle_vertices[1, 0]],
            [rectangle_vertices[0, 1], rectangle_vertices[1, 1]],
            color=color,
            label=file[:-5] 
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

plt.axis('equal')


plt.legend(title="id_data", bbox_to_anchor=(0.85, 1), loc='upper left')


plt.title('working face')

plt.tight_layout()
plt.show()
