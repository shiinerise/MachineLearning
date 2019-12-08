from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import metrics
digits = datasets.load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=666)

best_score = 0.0
best_k = 0
for k in range(1, 20):
    knn_classifier = KNeighborsClassifier(n_neighbors=k, weights="uniform", p=2)
    knn_classifier.fit(X_train, y_train)
    y_predict = knn_classifier.predict(X_test)
    score = metrics.accuracy(y_test, y_predict)
    if best_score < score:
        best_score = score
        best_k = k
# print("best_k:", best_k)
# print("best_score:", best_score)

param_search = [{"weights": ["uniform"], "n_neighbors": [i for i in range(1, 11)]},
    {"weights": ["distance"], "n_neighbors": [i for i in range(1, 11)], "p": [i for i in range(1, 6)]}]

knn_classifier = KNeighborsClassifier()
grid_search = GridSearchCV(knn_classifier, param_search)
grid_search.fit(X_train, y_train)
print("start")
print(grid_search.best_estimator_)