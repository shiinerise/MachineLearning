from sklearn import datasets
import numpy as np
from pyutil.model_selection import train_test_split
import matplotlib.pyplot as plt
from pyutil.LinearRegression import SimpleLinearRegression
import pyutil.metrics as mt
boston = datasets.load_boston()
X = boston.data[:, 5]
y = boston.target
X = X[y < 50]
y = y[y < 50]
X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
reg = SimpleLinearRegression()
reg.fit(X_train, y_train)
# print(reg._a)
# print(reg._b)
plt.scatter(X_train, y_train)
plt.plot(X_train, reg.predict(X_train), color='r')
plt.show()
y_predict = reg.predict(X_test)
# print(y_predict)
# print(y_predict.shape)
print("MSE", mt.mean_squared_error(y_test, y_predict))
print("RMSE", mt.root_mean_squared_error(y_test, y_predict))
print("MAE", mt.mean_absolute_error(y_test, y_predict))

