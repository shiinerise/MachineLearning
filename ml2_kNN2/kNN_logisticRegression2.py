from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
# 导入数据集
digits = datasets.load_digits()
print(digits.DESCR)
X = digits.data
y = digits.target.copy()
# 构造偏斜数据，将数字9的对应索引的元素设置为1，0～8设置为0
y[digits.target == 9] = 1
y[digits.target != 9] = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
# 使用逻辑回归做分类
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_predict = lr.predict(X_test)
matrix = confusion_matrix(y_test, y_predict)
decision_scores = lr.decision_function(X_test)
# P-R曲线
precision, recall, thresholds1 = precision_recall_curve(y_test, decision_scores)
plt.plot(precision, recall)
plt.show()
print(thresholds1)
# ROC曲线
fprs, tprs, thresholds2 = roc_curve(y_test, decision_scores)
plt.plot(fprs, tprs)
plt.show()
# AUC
print(roc_auc_score(y_test, decision_scores))







