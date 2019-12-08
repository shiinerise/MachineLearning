import numpy as np
from sklearn import datasets
from pyutil.LinearRegression import LinearRgression
from pyutil.model_selection import train_test_split
from sklearn import linear_model
import time
from pyutil import metrics
from sklearn.preprocessing import StandardScaler  # 数据归一化
boston = datasets.load_boston()
X = boston.data
y = boston.target
X = X[y<50]
y = y[y<50]
X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)
lin_reg1 = linear_model.LinearRegression()
print(time.asctime(time.localtime(time.time())))
lin_reg1.fit(X_train, y_train)
print(lin_reg1.score(X_test, y_test))
print(time.asctime(time.localtime(time.time())))

lin_reg2 = LinearRgression()
lin_reg2.fit_gredient(X_train, y_train, eta=0.00001)
X_test_plus = np.hstack([np.ones((len(X_test), 1)), X_test])
y_predict = X_test_plus.dot(lin_reg2._theta)
print(metrics.score(y_test, y_predict))
# print(lin_reg2.score(X_test, y_test, lin_reg2._theta))

standardScaler = StandardScaler()  # 数据归一化
standardScaler.fit(X_train)
X_train_std = standardScaler.transform(X_train)
X_test_std = standardScaler.transform(X_test)
lin_reg3 = LinearRgression()
lin_reg3.fit_gredient(X_train_std, y_train)
X_test_plus = np.hstack([np.ones((len(X_test), 1)), X_test_std])
y_predict = X_test_plus.dot(lin_reg3._theta)
print(metrics.score(y_test, y_predict))
# print(lin_reg3.score(X_test, y_test))
