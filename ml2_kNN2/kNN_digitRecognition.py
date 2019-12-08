import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import metrics
digits = datasets.load_digits()
X = digits.data
y = digits.target
img1 = X[555]
img1 = img1.reshape(8, 8)
plt.imshow(img1, cmap=matplotlib.cm.binary)
plt.show()
print(y[555])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
knn_classifier = KNeighborsClassifier(n_neighbors=50)
knn_classifier.fit(X_train, y_train)
y_predict = knn_classifier.predict(X_test)
# print("accuracy:", metrics.accuracy(y_test, y_predict))

best_score = 0.0
best_k = 0
for k in range(1, len(X_train) + 1):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    y_predict = knn_classifier.predict(X_test)
    score = metrics.accuracy(y_test, y_predict)
    if best_score < score:
        best_score = score
        best_k = k
print("best_k:", best_k)
print("best_score:", best_score)