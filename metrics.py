def accuracy(y_true, y_predict):
    assert y_true.shape[0] == y_predict.shape[0], \
        "the size of y_predict must be equal to y_true"
    return sum(y_true == y_predict) / len(y_true)