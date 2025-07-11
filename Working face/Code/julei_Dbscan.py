import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


data = pd.read_excel('../tezheng.xlsx')

xs = data['Out1']  
ys = data['Out2'] 


scaler = StandardScaler()
xs_scaled = scaler.fit_transform(xs.values.reshape(-1, 1))
ys_scaled = scaler.fit_transform(ys.values.reshape(-1, 1))

# Using DBSCAN clustering algorithm
dbscan = DBSCAN(eps=0.3, min_samples=5)  
dbscan.fit(xs_scaled)

# Obtain clustering labels
labels = dbscan.labels_.tolist() 

counts = np.zeros(max(labels) + 1)


for label in labels:
    counts[label] += 1

# Retrieve the index of the tag with the highest frequency of occurrence
most_common_cluster = np.argmax(counts)
print(most_common_cluster)


cluster_indices = [i for i, l in enumerate(labels) if l == most_common_cluster]

# Obtain trajectory data corresponding to the cluster label with the highest frequency
xs_cluster = np.array(xs[cluster_indices])
ys_cluster = np.array(ys[cluster_indices])


model = LinearRegression()
model.fit(xs_cluster.reshape(-1, 1), ys_cluster)

r2 = r2_score(ys_cluster, model.predict(xs_cluster.reshape(-1, 1)))
print(r2)

mae = mean_absolute_error(ys_cluster, model.predict(xs_cluster.reshape(-1, 1)))
print(mae)


offset = np.percentile(ys_cluster - model.predict(xs_cluster.reshape(-1, 1)), 5)


cluster_colors = {i: np.random.rand(3,) for i in set(labels)}

# Draw raw data, fitting curves, and boundary lines (above the fitting curve)
for label, color in cluster_colors.items():
    cluster_indices = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(xs[cluster_indices], ys[cluster_indices],label=f'Cluster {label}')

#plt.plot(xs_cluster, model.predict(xs_cluster.reshape(-1, 1)) - offset, color='green', linestyle='--', label='Boundary curve')
plt.legend()
plt.show()
