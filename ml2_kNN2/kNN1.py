import numpy as np
from sklearn import datasets
# from pyutil.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import ML_algorithms
iris = datasets.load_iris()
X = iris.data
y = iris.target
# print(X.shape)
# print(y.shape)
# dataConcat = np.concatenate((X, y.reshape(-1, 1)), axis=1)
# np.random.shuffle(dataConcat)
# # print(dataConcat)
# X_shuffle, y_shuffle = np.split(dataConcat, [4], axis=1)
# test_rate = 0.2
# test_size = int(len(X) * test_rate)
# X_train = X_shuffle[test_size:]
# X_test = X_shuffle[:test_size]
# y_train = y_shuffle[test_size:]
# y_test = y_shuffle[:test_size]
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=500)
# print(X_train.shape[0])
# print(X_test.shape)
# print(y_train.shape[0])
# print(y_test.shape)
mykNNClassifier = ML_algorithms.KNNClassifier(k=6)
mykNNClassifier.fit(X_train, y_train)
y_predict = mykNNClassifier.predict(X_test)
print("accuracy:", np.count_nonzero((y_predict == y_test) == True)/len(y_test))
