import numpy as np
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
boston = datasets.load_boston()
x = boston.data[:, 5]
y = boston.target
# print(boston.keys())
# print(boston.feature_names)
print(x.shape)
# print(y.shape)
# print(boston.DESCR)
# plt.scatter(x, y)
# plt.show()
x = x[y < 50]
y = y[y < 50]
# plt.scatter(x, y)
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=666)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
linear_regression = LinearRegression()
linear_regression.fit(x_train.reshape(1, -1), y_train)
y_predict = linear_regression.predict(x_train)
plt.scatter(x_train, y_train)
plt.plot(x_train, y_predict, color='r')
plt.show()