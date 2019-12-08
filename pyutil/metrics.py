import numpy as np
from math import sqrt
def accuracy(y_true, y_predict):
    assert y_true.shape[0] == y_predict.shape[0], \
        "the size of y_predict must be equal to y_true"
    return sum(y_true == y_predict) / len(y_true)
def mean_squared_error(y_true, y_predict):
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to y_predict"
    return np.sum((y_true-y_predict)**2)/len(y_true)
def root_mean_squared_error(y_true, y_predict):
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to y_predict"
    return sqrt(np.sum((y_true-y_predict)**2)/len(y_true))
def mean_absolute_error(y_true, y_predict):
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to y_predict"
    return np.sum(np.absolute(y_true - y_predict))/len(y_true)
def score(y_true, y_predict):
    return r2_score(y_true, y_predict)

def r2_score(y_true, y_predict):
    return 1 - mean_squared_error(y_true, y_predict)/np.var(y_true)
