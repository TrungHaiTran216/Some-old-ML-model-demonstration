import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

means = [[2, 2], [8, 3], [3, 6],[10,8]]
cov = [[0.8, 0], [0, 0.8]]
N = 100
np.random.seed(10)

# Data set up
X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T
X2 = np.random.multivariate_normal(means[2], cov, N).T
X3 = np.random.multivariate_normal(means[3], cov, N).T
X = np.concatenate((X0,X1,X2,X3),axis=1)
X = np.vstack((X,np.ones((1,4*N))))
total = 4*N
y0 = np.ones((1,N))*0
y1 = np.ones((1,N))*1
y2 = np.ones((1,N))*2
y3 = np.ones((1,N))*3

yy = np.concatenate((y0,y1,y2,y3),axis = 1)
yy = yy.astype(int)
y = np.eye(np.max(yy[0,:]) + 1)[yy[0,:]].T

# Softmax regression main function

def loss(W):
    Z = W @ X
    Z = Z - np.max(Z, axis=0)
    expZ = np.exp(Z)
    Yhat = expZ / np.sum(expZ, axis=0)
    loss = -np.sum(y * np.log(Yhat)) / X.shape[1]
    return loss

def softmax(z):
    z = z - np.max(z)
    expz = np.exp(z)
    return expz / np.sum(expz)

def GD(W,eta):
    W_history = []
    for j in range(100):
        W_copy = W.copy()
        for i in range(X.shape[1]):
            W = W - eta*gradient(W,i)
        if j % 5 == 0:
            W_history.append(W)
        if abs(loss(W_copy) - loss(W)) < 1e-5:
            return W,W_history
    return W,W_history
        
def gradient(W,k):
    z_k = W @ X[:,k]
    yhat_k = softmax(z_k)
    y_k = y[:,k]
    grad = np.outer(yhat_k - y_k, X[:,k])
    return grad

def predict(W, X):
    Z = W @ X
    Yhat = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return np.argmax(Yhat, axis=0)

# Softmax regression test

W_init = 0.01 * np.random.randn(4,3)
W,W_history = GD(W_init,0.01)
y_pred = predict(W, X)
accuracy = np.mean(y_pred == yy[0])
print("Accuracy:", accuracy)

# Demonstrate

def decisionBoundary(W,X):
    x_min, x_max = X[0,:].min() - 1, X[0,:].max() + 1
    y_min, y_max = X[1,:].min() - 1, X[1,:].max() + 1
    xxx, yyy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    grid = np.vstack((xxx.ravel(), yyy.ravel(), np.ones(xxx.size)))
    Z = W @ grid
    Yhat = softmax(Z)
    classes = np.argmax(Yhat, axis=0)
    classes = classes.reshape(xxx.shape)
    plt.contourf(xxx, yyy, classes, alpha=0.3, cmap='viridis')
    
#decisionBoundary(W,X)

def update(frame):
    plt.cla()
    plt.scatter(X0[0,:],X0[1,:], marker='o', alpha=None)
    plt.scatter(X1[0,:],X1[1,:], marker='v', alpha=None)
    plt.scatter(X2[0,:],X2[1,:], marker='*', alpha=None)
    plt.scatter(X3[0,:],X3[1,:], marker='D', alpha=None)
    W = W_history[frame]
    decisionBoundary(W, X)
    plt.title(f"Step {frame}")

fig = plt.figure()

anim = FuncAnimation(
    fig,
    update,
    frames=len(W_history),
    interval=100,
    repeat=True
)

plt.show()