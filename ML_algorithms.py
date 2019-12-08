import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
class KNNClassifier:
    def __init__(self, k):
        assert k >= 1, "k must be valid"
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert X_train.shape[0] >= self.k, "the size of X_train must be at least k"

        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        assert self._X_train is not None and self._y_train is not None, "X_train and y_train must be not none"
        assert self._X_train.shape[1] == X_predict.shape[1], "the feature number of X_train must be equal to X_predict"
        y_predict = [self._predict(x_predict) for x_predict in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        distances = [np.sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        nearest = np.argsort(distances)[:self.k]
        count = Counter(self._y_train[nearest])
        return count.most_common(1)[0][0]

    def __repr__(self):
        return "knn(k=%d)" % self.k