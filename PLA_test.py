import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

means = [[2,2],[4,2]]
cov = [[0.3, 0.2],[0.2,0.3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T

X = np.concatenate((X0,X1), axis = 1)
X = np.vstack((X,np.ones((1,X.shape[1]))))

y = np.concatenate((np.ones(N),-np.ones(N)))

w  = np.random.randn(X.shape[0])

print(X.shape)
print(y.shape)
print(w.shape)
def updatew(w):
    for j in range(0,100):
        history = []
        mask = y * (w @ X) <= 0
        X_mis = X[:,mask]
        y_mis = y[mask]    
        for i in range(X_mis.shape[1]):
            w += 0.5 * y_mis[i]*X_mis[:,i]
            history.append(w.copy())
        if np.array_equal(np.sign(w.T @ X), y):
            return w, history
    return w, history
w, history = updatew(w)
X3 = np.random.multivariate_normal(means[0], cov, 1).T

from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()

ax.scatter(X0[0,:],X0[1,:],color='blue',marker='D')
ax.scatter(X1[0,:],X1[1,:],color='red',marker='o')

line, = ax.plot([],[],color='black')
print(len(history))
def update(frame):
    w = history[frame]

    w1,w2,w3 = w

    x = np.linspace(-1,6,100)
    y = -(w1*x + w3)/w2

    line.set_data(x,y)
    return line,

ani = FuncAnimation(fig, update, frames=len(history), interval=300)

plt.show()