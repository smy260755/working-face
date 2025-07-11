import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


data = pd.read_excel('../data/604_15th.xlsx')

# Extract Out1 and Out2 columns as features
X = data[['Out1', 'Out2']].values


k = 4

# Calculate K-distance
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)
k_distances = distances[:, k-1]


k_distances = np.sort(k_distances)

# Draw K-distance diagram
plt.plot(k_distances)
plt.title("K-dist Plot")
plt.xlabel("Points")
plt.ylabel("K-distance")
plt.show()
