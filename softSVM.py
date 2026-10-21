import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from cvxopt import matrix, solvers

means = [[2, 2], [4, 1]]
cov = [[.3, .2], [.2, .3]]
N = 10
np.random.seed(20)
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X1[-1, :] = [2.7, 2]
X = np.concatenate((X0.T, X1.T), axis = 1)
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1)
V = X * y
P = matrix(V.T @ V)

q = matrix(-np.ones((2*N,1)))
c = 1e-2
G = matrix(np.vstack((-np.eye(2*N),np.eye(2*N))))
h = matrix(np.vstack((np.zeros((1,2*N)).T,np.ones((1,2*N)).T*c)))
print(h)
A = matrix(y)
b = matrix(np.zeros((1,1)))

solvers.options['show_progress'] = False
sol = solvers.qp(P,q,G,h,A,b)
l = np.array(sol['x'])
print('lambda = ')
print(l.T)

epsilon = 1e-5
pos = np.where(l > epsilon)[0]
lS = l[pos]
VS = V[:,pos]
w = VS @ lS
b = np.mean(y[0,pos]- (w.T @ X[:,pos]))

def plot_svm(w, b, X, y):
    plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', alpha=0.7)
    x_vals = np.linspace(plt.xlim()[0], plt.xlim()[1], 50)
    y_vals = -(w[0] * x_vals + b) / w[1]    
    y_vals_pos = -(w[0] * x_vals + b - 1) / w[1]
    y_vals_neg = -(w[0] * x_vals + b + 1) / w[1]
    plt.plot(x_vals, y_vals, 'k-', label='decision boundary')
    plt.plot(x_vals, y_vals_pos, 'k--', label='+margin')
    plt.plot(x_vals, y_vals_neg, 'k--', label='-margin')
    plt.show()
    
plot_svm(w,b,X.T,y)