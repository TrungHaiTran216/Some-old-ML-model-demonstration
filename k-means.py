# ===== Core numerical computing =====
import numpy as np
import scipy as sp

# ===== Data handling =====
import pandas as pd

# ===== Visualization =====
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

k = 3
rng = np.random.default_rng(10)
centroid = np.array([rng.uniform(-10, 10, size=2) for _ in range(k)])
point = []
for c in centroid:
    pts = rng.normal(c,1,size=(100,2))
    point.append(pts)
points = np.concatenate([d for d in point], axis=0)
ini_centroid = points[rng.choice(len(points), size=k, replace=False)]
#plt.plot(point[0][:, 0], point[0][:, 1], 'b^', markersize = 4, alpha = .8)
#plt.plot(point[1][:, 0], point[1][:, 1], 'go', markersize = 4, alpha = .8)
#plt.plot(point[2][:, 0], point[2][:, 1], 'rs', markersize = 4, alpha = .8)
plt.axis('equal')
plt.grid(True)
def assign(points, center):
    distances = np.linalg.norm(points[:, None, :] - center[None, :, :], axis=2)
    labels = np.argmin(distances, axis=1)
    return labels
def update_center(points,labels,k):
    center = np.zeros((k,2))
    for j in range(k):
        cluster_points = points[labels == j]
        center[j] = cluster_points.mean(axis=0)
    return center
        
def kmeans(points, ini_centroid, k):
    centroids = [ini_centroid]
    for it in range(1000):
        new_center = update_center(points,assign(points,centroids[-1]),k)
        #plt.plot(new_center[:,0],new_center[:,1],'ko',markersize = 5, alpha = 1)
        if np.allclose(new_center, centroids[-1]):
            break
        centroids.append(new_center)
    return new_center
res = kmeans(points,ini_centroid,k)
labels = assign(points,res)
plt.scatter(points[:,0], points[:,1],c=labels,cmap='viridis')
plt.plot(res[:,0],res[:,1],'co',markersize = 5, alpha = 1)
#plt.plot(ini_centroid[:,0],ini_centroid[:,1],'co',markersize = 5, alpha = 1)
plt.show()