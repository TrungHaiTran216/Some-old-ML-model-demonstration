import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from cvxopt import matrix, solvers

np.random.seed(5)
means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N) # class 1
X1 = np.random.multivariate_normal(means[1], cov, N) # class -1 
X = np.concatenate((X0.T, X1.T), axis = 1) # all data 
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1) 
fig = plt.figure()
plt.scatter(X0.T[0,:],X0.T[1,:],marker='o',c ='red')
plt.scatter(X1.T[0,:],X1.T[1,:],marker='^',c ='blue')
plt.show()
V = X * y
P = np.outer(y.flatten(), y.flatten()) * (X.T @ X)
P = np.array(P, dtype=np.float64)
P = matrix(P)
q = matrix(-np.ones((2*N,1)))

A = matrix(y)

b = matrix(np.zeros((1,1)))

G = matrix(-np.eye(2*N))

h = matrix(np.zeros((2*N,1)))

solvers.options['show_progress'] = False
sol = solvers.qp(P,q,G,h,A,b)
l = np.array(sol['x'])
print('lambda = ')
print(l.T)

epsilon = 1e-6
S = np.where(l > epsilon)[0]

VS = V[:, S]
XS = X[:, S]
yS = y[:, S]
lS = l[S]
print(VS.shape)
print(XS.shape)
print(yS.shape)
print(lS.shape)
# calculate w and b
w = VS.dot(lS)
b = np.mean(yS.T - w.T.dot(XS))

print('w = ', w.T)
print('b = ', b)