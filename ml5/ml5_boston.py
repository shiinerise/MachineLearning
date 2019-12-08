from sklearn import datasets
from pyutil.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pyutil.LinearRegression
from sklearn.linear_model import SGDRegressor
boston = datasets.load_boston()
X = boston.data
y = boston.target
X = X[y<50]
y = y[y<50]
X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)
standardScaler = StandardScaler()
standardScaler.fit(X_train)
X_train_std = standardScaler.transform(X_train)
X_test_std = standardScaler.transform(X_test)
# lin_reg1 = pyutil.LinearRegression.LinearRegression()
# lin_reg1._theta = lin_reg1.fit_SGD(X_train, y_train, n_iters=2)
# lin_reg1.score(X_test, y_test, lin_reg1._theta)
sgd_reg = SGDRegressor(n_iter=100)
sgd_reg.fit(X_train_std, y_train)
print(sgd_reg.score(X_test_std, y_test))



