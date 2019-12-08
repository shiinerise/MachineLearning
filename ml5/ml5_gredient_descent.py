import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative
def loss_function(x):
    try:
        return (x-2.5) ** 2 - 1
    except:
        return float('inf')
plot_x = np.linspace(-1, 6, 140)
plot_y = loss_function(plot_x)
def derivative_loss_function(theta):
    return derivative(loss_function, theta, dx=1e-6)

thetaList = []
def gradient_descent(initial_theta=0, eta=0.1, n_iters=10000, epsilon=1e-6):
    theta = initial_theta
    thetaList.append(theta)
    i_iters = 0
    while i_iters < n_iters:
        # 梯度
        gradient = derivative_loss_function(theta)
        last_theta = theta
        theta = theta - eta * gradient
        thetaList.append(theta)
        if(abs(loss_function(theta) - loss_function(last_theta)) < epsilon):
            break
        i_iters += 1

def plot_theta_history():
    plt.plot(plot_x, plot_y)
    plt.plot(np.array(thetaList), loss_function(np.array(thetaList)), color='red', marker='o')
    plt.show()

gradient_descent(eta=0.1)
plot_theta_history()
print("1梯度下降查找次数：", len(thetaList))
gradient_descent(eta=0.01)
plot_theta_history()
print("2梯度下降查找次数：", len(thetaList))
gradient_descent(eta=0.8)
plot_theta_history()
print("3梯度下降查找次数：", len(thetaList))

