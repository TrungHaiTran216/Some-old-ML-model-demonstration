import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split


# load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# basic info
print('Number of classes:', len(np.unique(y)))
print('Number of data points:', len(y))

# split by class (boolean indexing)
X0 = X[y == 0]
X1 = X[y == 1]
X2 = X[y == 2]

# show samples
#print('\nSamples from class 0:\n', X0[:5])
#print('\nSamples from class 1:\n', X1[:5])
#print('\nSamples from class 2:\n', X2[:5])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50, random_state=42)
clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('\nTrue labels:\n', y_test)
print('\nPredicted labels:\n', y_pred)