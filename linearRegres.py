# ===== Core numerical computing =====
import numpy as np
import scipy as sp

# ===== Data handling =====
import pandas as pd

# ===== Visualization =====
import matplotlib.pyplot as plt
import seaborn as sns


rng = np.random.default_rng(42)
X = rng.random((1000, 1))
one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one,X), axis = 1)
y = 4 + 3 * X + .2*rng.standard_normal((1000, 1))
def cost(w):
    N = X.shape[0]
    return 0.5/N*np.linalg.norm(y - Xbar.dot(w),2)**2
def numerical_grad(w):
    eps = 1e-4
    g = np.zeros_like(w)
    for i in range(len(w)):
        w_p = w.copy()
        w_n = w.copy()
        w_p[i] += eps
        w_n[i] -= eps
        g[i] = (cost(w_p)-cost(w_n))/(2*eps)
    return g
def myGD(w_initial,eta):
    W = [w_initial]
    for it in range(100):
        w_new = W[-1] - eta*numerical_grad(W[-1])
        W.append(w_new)
        if(cost(w_new) < 1e-3):
        #if np.linalg.norm(numerical_grad(w_new))/len(w_new) < 1e-3:
            break
    return (w_new, it)
eta = 1
w_initial = np.array([[2.], [1.]])
(w1, it1) = myGD(w_initial,eta)
print(w1, it1)
