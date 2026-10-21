# ===== Core numerical computing =====
import numpy as np
import scipy as sp

# ===== Data handling =====
import pandas as pd

# ===== Visualization =====
import matplotlib.pyplot as plt
import seaborn as sns

k = 3
vote = 10
rng = np.random.default_rng(17)
centroid = np.array([rng.uniform(-10, 10, size=2) for _ in range(k)])
point = []
labels = []
for i in range(len(centroid)):
    pts = rng.normal(centroid[i],1,size=(100,2))
    labels += [i]*100
    point.append(pts)
points = np.concatenate([d for d in point], axis=0)
rng2 = np.random.default_rng(20)
testing = np.array([rng.uniform(-10, 10, size=2) for _ in range(8)])
plt.scatter(points[:,0], points[:,1],c=labels,cmap='viridis')
distances = np.zeros((len(testing),len(points)))
knn_labels = np.zeros(len(testing), dtype=int)
knn_idx = np.zeros((len(testing), vote), dtype=int)

for i in range(len(testing)):
    for j in range(len(points)):
        distances[i, j] = np.linalg.norm(testing[i] - points[j])
    knn_idx[i] = np.argsort(distances[i])[:vote]
    knn_labels[i] = np.bincount(np.array(labels)[knn_idx[i]]).argmax()
plt.scatter(testing[:,0], testing[:,1],c=knn_labels,cmap='cool', marker='X', s=200)
plt.grid(True)
plt.show()