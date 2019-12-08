import numpy as np
import matplotlib.pyplot as plt
import time
from pyutil import LinearRegression
m = 10000
x = np.random.normal(size=m)  # 正态分布
X = x.reshape(-1, 1)
# print(X)
y = 4. * x + 3. + np.random.normal(0, 3, size=m)

def J(theta, X_b, y):
    try:
        return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
    except:
        return float('inf')  # 返回无穷大

def dJ(theta, X_b, y):
    return X_b.T.dot(X_b.dot(theta) - y) * 2./len(y)


def gredient_descent(X_b, y, inital_theta, eta=0.1, n_iters=1e4, epsilon=1e-8):
    theta = inital_theta
    i_iters = 0
    while i_iters < n_iters:
        gradient = dJ(theta, X_b, y)
        last_theta = theta
        theta = theta - eta * gradient
        if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
            break
        i_iters += 1
    return theta
print(time.asctime(time.localtime(time.time())))
X_b = np.hstack([np.ones((len(X), 1)), X])
initial_theta = np.zeros(X_b.shape[1])
eta = 0.01
theta = gredient_descent(X_b, y, initial_theta, eta)
print(theta)
print(time.asctime(time.localtime(time.time())))

def dJ_SGD(theta, X_b_i, y_i):
    return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2
def sgd(X_b, y, initial_theta, n_iters):
    t0 = 5
    t1 = 50
    def learning_rate(cur_iter):
        return t0/(t1+cur_iter)
    theta = initial_theta
    for cur_iter in range(n_iters):
        # 随机找到一个样本，得到其索引
        rand_i = np.random.randint(len(X_b))
        gradient = dJ_SGD(theta, X_b[rand_i], y[rand_i])
        theta = theta - learning_rate(cur_iter) * gradient
    return theta
print(time.asctime(time.localtime(time.time())))
X_b = np.hstack([np.ones((len(X), 1)), X])
initial_theta = np.zeros(X_b.shape[1])
eta = 0.01
theta = sgd(X_b, y, initial_theta, n_iters=len(X_b)//3)
print(theta)
print(time.asctime(time.localtime(time.time())))