import numpy as np
class LinearRegression:
    def __init__(self):
        self._a = None
        self._b = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, "简单线性回归仅能处理一维特征向量"
        assert len(x_train) == len(y_train), "特征向量的长度和标签的长度必须相等"
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)
        self._a = num / d
        self._b = y_mean - self._a * x_mean
        return self

    def predict(self, x_single):
        """
        预测y_predict
        :param x_single: 给定的单个待测数据
        :return:y_predict
        """
        y_predict = x_single * self._a + self._b
        return y_predict

    def __repr__(self):
        """返回一个可以用来表示对象的可打印的字符串"""
        return "LinearRegression()"
    def fit_gredient(self, X_train, y_train, eta=0.1, n_iters=1e4):
        assert X_train.shape[0] == y_train.shape[0], "the shape of X_train must be equals to y_train"
        def J(theta, X_b, y):
            try:
                return np.sum((y-X_b.dot(theta))**2)/len(y)
            except:
                return float('inf')  # 返回无穷大

        def dJ(theta, X_b, y):
            return X_b.T.dot(X_b.dot(theta) - y) * 2./len(y)

        def gredient_descent(X_b, y, inital_theta, eta=0.1, n_iters=1e4, epsilon=1e-8):
            theta = inital_theta
            i_iters = 0
            while i_iters<n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if(abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break
                i_iters += 1
            return theta
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        inital_theta = np.zeros(X_b.shape[1])
        self._theta = gredient_descent(X_b, y_train, inital_theta, eta, n_iters)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def score(self, X_test, y_test, _theta):
        X_test_plus = np.hstack([np.ones((len(X_test), 1)), X_test])
        return np.sum((y_test-X_test_plus.dot(_theta))**2)/len(y_test)

    def fit_SGD(self, X_train, y_train, n_iters=50, t0=5, t1=50):
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to y_train"
        assert n_iters >= 1

        def dJ(theta, X_b_i, y_i):
            return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2
        def sgd(X_b, y, initial_theta, n_iters=50, t0=5, t1=50):
            def learning_rate(cur_iter):
                return t0/(t1+cur_iter)
            theta = initial_theta
            m = len(X_b)
            for cur_iter in range(n_iters):
                indexes = np.random.permutation(m)  # 将原本的数据随机打乱
                X_b_new = X_b[indexes, :]
                y_new = y[indexes]
                for i in range(m):
                    gradient = dJ(theta, X_b_new[i], y_new[i])
                    theta = theta - learning_rate(cur_iter * m + i) * gradient
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        inital_theta = np.random.randn(X_b.shape[1])
        self._theta = sgd(X_b, y_train, inital_theta, n_iters, t0, t1)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self



