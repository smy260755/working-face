import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from scipy.spatial import ConvexHull

data_folder = '../data/'

working_face_files = [f'639_{i:02d}th.xlsx' for i in range(10, 16)]  
tunnel_files = [f'639_{i:02d}th.xlsx' for i in range(3, 28)] 

colors = plt.cm.tab10(np.linspace(0, 1, len(working_face_files)))

plt.figure(figsize=(12, 8))

for file, color in zip(working_face_files, colors):
    try:
        data = pd.read_excel(os.path.join(data_folder, file))
        xs = data['Out1'].values.reshape(-1, 1)
        ys = data['Out2'].values.reshape(-1, 1)


        dbscan = DBSCAN(eps=0.3, min_samples=100)
        dbscan.fit(np.column_stack((xs, ys)))

        labels = dbscan.labels_
        cluster_indices = [i for i, l in enumerate(labels) if l != -1]

        xs_cluster = xs[cluster_indices]
        ys_cluster = ys[cluster_indices]

        data_pca = np.column_stack((xs_cluster, ys_cluster))


        pca = PCA(n_components=2)
        pca.fit(data_pca)

        projected_data = data_pca @ pca.components_.T

        # Calculate the boundary containing 90% of the data points
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


        plt.plot([rectangle_vertices[0, 0], rectangle_vertices[1, 0]],
                 [rectangle_vertices[0, 1], rectangle_vertices[1, 1]], color=color, label=file[:-5])
        plt.plot([rectangle_vertices[1, 0], rectangle_vertices[2, 0]],
                 [rectangle_vertices[1, 1], rectangle_vertices[2, 1]], color=color)
        plt.plot([rectangle_vertices[2, 0], rectangle_vertices[3, 0]],
                 [rectangle_vertices[2, 1], rectangle_vertices[3, 1]], color=color)
        plt.plot([rectangle_vertices[3, 0], rectangle_vertices[0, 0]],
                 [rectangle_vertices[3, 1], rectangle_vertices[0, 1]], color=color)

    except Exception as e:
        print("error")

# Collect all original points to draw tunnel shapes
all_tunnel_points = []
for file in tunnel_files:
    try:
        data = pd.read_excel(os.path.join(data_folder, file))
        xs = data['Out1'].values.reshape(-1, 1)
        ys = data['Out2'].values.reshape(-1, 1)
        all_tunnel_points.append(np.column_stack((xs, ys)))

    except Exception as e:
        print("error")

# Draw the shape of the tunnel
if all_tunnel_points:
    all_tunnel_points = np.vstack(all_tunnel_points)
    hull = ConvexHull(all_tunnel_points)
    hull_vertices = all_tunnel_points[hull.vertices]

    plt.plot(np.append(hull_vertices[:, 0], hull_vertices[0, 0]),
             np.append(hull_vertices[:, 1], hull_vertices[0, 1]),
             'k-', linewidth=2, label='Tunnel')


plt.axis('equal')


plt.legend(title="Legend", bbox_to_anchor=(0.85, 1), loc='upper left')


plt.title('Working Face and Tunnel')


plt.tight_layout()
plt.show()
