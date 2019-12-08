import numpy as np
def train_test_split(X, y, test_rate=0.2, seed=None):
    """
    分割训练集和测试集的方法
    :param X:features
    :param y:labels
    :param test_rate:测试数据占总数据集的概率
    :param seed:随机数种子
    :return:划分好的训练集和测试集的features和labels
    """
    assert X.shape[0] == y.shape[0], 'the size of X must be equals to y'
    assert 0.0 < test_rate < 1.0, 'test_rate must be in (0.0, 1.0)'
    if seed:
        np.random.seed(seed)
    shuffle_index = np.random.permutation(len(X))
    test_size = int(len(X) * test_rate)
    X_train = X[shuffle_index[test_size:]]
    X_test = X[shuffle_index[:test_size]]
    y_train = y[shuffle_index[test_size:]]
    y_test = y[shuffle_index[:test_size]]
    return X_train, X_test, y_train, y_test