import os, warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", message=".*np.object.*", category=FutureWarning)
# ===== Core numerical computing =====
import numpy as np

# ===== Data handling =====

# ===== Visualization =====
from sklearn.cluster import KMeans
from scipy.stats import mode
import tensorflow as tf
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

#X_train = X_train[:500]
#y_train = y_train[:500]

X_train = X_train.reshape(X_train.shape[0],-1)
X_train = X_train.astype(np.float32) / 255.0 
model = KMeans(n_clusters=10,n_init=10,random_state=0,max_iter=50).fit(X_train)

digit_of_cluster = {}
labels = model.labels_
for c in range(10):
    loc = (labels == c)
    represent = mode(y_train[loc])
    digit_of_cluster[c] = represent.mode

pred = np.array([digit_of_cluster[c] for c in labels])

accuracy = np.mean(pred == y_train)

print(accuracy)

centers = model.cluster_centers_
centers = centers.reshape(10,28,28)
centers = (centers*255.0).astype(np.uint8)

fig, axes = plt.subplots(2, 5, figsize=(10, 4))

for i, ax in enumerate(axes.flat):
    ax.imshow(centers[i], cmap='gray')
    ax.set_title(f'Cluster {digit_of_cluster[i]}')
    ax.axis('off')

plt.tight_layout()
plt.show()



