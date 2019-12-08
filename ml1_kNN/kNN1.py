from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter #投票
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
# 导入鸢尾花数据集
iris = datasets.load_iris()
# 查看数据
print(iris)
X = iris.data
y = iris.target
# 数据可视化
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='r', label='1')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='g', label='2')
plt.scatter(X[y == 2][:, 0], X[y == 2][:, 1], color='b', label='3')

plt.show()
# 随机划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20, shuffle=True)

# 计算距离
def calculateDistance(data1, data2):
    # 欧式距离
    dist = np.sqrt(sum((data1 - data2) ** 2))
    return dist

# knn模型分类
def knnClassify(X_train, y_train, test_data, k):
    dist = [calculateDistance(train_data, test_data) for train_data in X_train]
    # 返回距离最近的k个训练样本的索引（下标）
    indexes = np.argsort(dist)[:k]
    count = Counter(y_train[indexes])
    return count.most_common(1)[0][0]

# 预测结果
predictions = [knnClassify(X_train, y_train, test_data, 3) for test_data in X_test]
# 与实际结果对比
correct = np.count_nonzero((predictions == y_test) == True)
print("Accuracy is: %.3f" % (correct/len(X_test)))

# from sklearn.model_selection import KFold
# kf = KFold(n_splits=2)
# for train_index, test_index in kf.split(X):
#     print('X_train:%s ' % X[train_index])
#     print('X_test: %s ' % X[test_index])
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     # 创建kNN_classifier实例
#     kNN_classifier = KNeighborsClassifier(n_neighbors=3)
#     # kNN_classifier做一遍fit(拟合)的过程，没有返回值，模型就存储在kNN_classifier实例中
#     kNN_classifier.fit(X_train, y_train)
#     correct = np.count_nonzero((kNN_classifier.predict(X_test) == y_test) == True)
#     print("Accuracy is: %.3f" % (correct/len(X_test)))