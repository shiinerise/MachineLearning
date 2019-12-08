from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter #投票
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
# 导入鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target
# 随机划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20, shuffle=True)
# 数据可视化
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='r')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='g')
plt.scatter(X_train[y_train == 2][:, 0], X_train[y_train == 2][:, 1], color='b')
plt.scatter(6.1, 3.2, color='black')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()

# 计算距离，默认为欧氏距离
def calculateDistance(data1, data2, p=2):
    if len(data1) == len(data2) and len(data1) >= 1:
        sum = 0
        for i in range(len(data1)):
            sum += math.pow(abs(data1[i] - data2[i]), p)
            dist = math.pow(sum, 1/p)
    return dist

# knn模型分类
def knnClassify(X_train, y_train, test_data, k):
    dist = [calculateDistance(train_data, test_data) for train_data in X_train]
    # 返回距离最近的k个训练样本的索引（下标）
    indexes = np.argsort(dist)[:k]
    count = Counter(y_train[indexes])
    return count.most_common(1)[0][0]

if __name__ == '__main__':
    # 预测结果
    predictions = [knnClassify(X_train, y_train, test_data, 3) for test_data in X_test]
    # 与实际结果对比
    correct = np.count_nonzero((predictions == y_test) == True)
    print("Accuracy is: %.3f" % (correct/len(X_test)))

# 创建kNN_classifier实例
kNN_classifier = KNeighborsClassifier(n_neighbors=3)
# kNN_classifier做一遍fit(拟合)的过程，没有返回值，模型就存储在kNN_classifier实例中
kNN_classifier.fit(X_train, y_train)
# kNN进行预测predict，需要传入一个矩阵，而不能是一个数组。
# reshape()成一个二维数组，第一个参数是1表示只有一个数据，第二个参数-1，numpy自动决定第二维度有多少
correct = np.count_nonzero((kNN_classifier.predict(X_test) == y_test) == True)
print("Accuracy is: %.3f" % (correct/len(X_test)))