import sklearn
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from metrics import accuracy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

digits = datasets.load_digits()
# print(digits.keys())
X = digits.data
y = digits.target.copy()
y[digits.target == 9] = 1
y[digits.target != 9] = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_predict = lr.predict(X_test)
print(accuracy(y_test, y_predict))
def TN(y_true, y_predict):
    assert len(y_true) == len(y_predict), "the size of y_true must be equal to y_predict"
    return np.sum((y_true == 0) & (y_predict == 0))
# print(TN(y_test, y_predict))
def TP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 1))
def FN(y_true, y_predict):
    assert len(y_true) == len(y_predict), "the size of y_true must be equal to y_predict"
    return np.sum((y_true == 1) & (y_predict == 0))
def FP(y_true, y_predict):
    assert len(y_true) == len(y_predict), "the size of y_true must be equal to y_predict"
    return np.sum((y_true == 0) & (y_predict == 1))
def confusion_matrix(y_true, y_predict):
    return np.array([[TN(y_true, y_predict), FP(y_true, y_predict)],
                     [FN(y_true, y_predict), TP(y_true, y_predict)]])
print(confusion_matrix(y_test, y_predict))
def precision_score(y_true, y_predict):
    matrix = confusion_matrix(y_true, y_predict)
    return matrix[1][1] / (matrix[1][1] + matrix[0][1])
def recall_score(y_true, y_predict):
    matrix = confusion_matrix(y_true, y_predict)
    return matrix[1][1] / (matrix[1][1] + matrix[1][0])
# print(precision_score(y_test, y_predict))
# print(recall_score(y_test, y_predict))

matrix = confusion_matrix(y_test, y_predict)
# print(matrix)

def f1_score(precision, recall):
    return 2 * precision * recall / (precision + recall)
# print(f1_score(0.5, 0.5))
# print(f1_score(0.9, 0.1))
decision_scores = lr.decision_function(X_test)
# print("decision_score", decision_scores)

def TPR(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return tp / (tp+fn)
    except:
        return 0.0

def FPR(y_true, y_predict):
    fp = FP(y_true, y_predict)
    tn = TN(y_true, y_predict)
    try:
        return fp / (tn+fp)
    except:
        return 0.0
fprs = []
tprs = []
# 创建阈值的集合
thresholds = np.arange(np.min(decision_scores), np.max(decision_scores), 0.1)
for threshold in thresholds:
    # 设定阈值，把大于阈值的分类为1，小于阈值的分类为0
    y_predict = np.array(decision_scores > threshold, dtype=int)
    fprs.append(FPR(y_test, y_predict))
    tprs.append(TPR(y_test, y_predict))
plt.plot(fprs, tprs)
plt.xlabel('fprs')
plt.ylabel('tprs')
plt.show()