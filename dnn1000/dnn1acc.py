import pandas as pd
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error, roc_curve, classification_report,auc)


testdata = pd.read_csv('dnnres/dnn1predicted.txt', header=None)
traindata = pd.read_csv('dnnres/expected.txt', header=None)


y_train1 = traindata
y_pred = testdata
accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")

print("dnn1results")
print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)

print("/n")

testdata = pd.read_csv('dnnres/dnn2predicted.txt', header=None)
traindata = pd.read_csv('dnnres/expected.txt', header=None)


y_train1 = traindata
y_pred = testdata
accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")

print("dnn2results")
print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)
print("/n" )


testdata = pd.read_csv('dnnres/dnn3predicted.txt', header=None)
traindata = pd.read_csv('dnnres/expected.txt', header=None)


y_train1 = traindata
y_pred = testdata
accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")

print("dnn3results")
print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)
print("/n" )


testdata = pd.read_csv('dnnres/dnn4predicted.txt', header=None)
traindata = pd.read_csv('dnnres/expected.txt', header=None)


y_train1 = traindata
y_pred = testdata
accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")

print("dnn4results")
print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)
print("/n" )

testdata = pd.read_csv('dnnres/dnn5predicted.txt', header=None)
traindata = pd.read_csv('dnnres/expected.txt', header=None)


y_train1 = traindata
y_pred = testdata
accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")

print("dnn5results")
print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)
print("/n" )

