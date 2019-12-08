import numpy as np
import matplotlib.pyplot as plt
from pyutil.LinearRegression import SimpleLinearRegression
x = np.array([1., 2., 3., 4., 5.])
y = np.array([1., 3., 2., 3., 5.])
reg = SimpleLinearRegression()
reg.fit(x, y)

y_hat = reg._a * x + reg._b
plt.scatter(x, y)
plt.axis([0, 6, 0, 6])
plt.plot(x, y_hat, color='r')
plt.show()

x_predict = np.array([6])
y_predict = reg.predict(x_predict)
print(y_predict)