
# coding: utf-8

import pandas as pd
import os
import numpy as np
from sklearn import svm


df = pd.read_csv("~/Desktop/glassType.txt")
df = df.iloc[np.random.permutation(len(df))]
df_norm = (df - df.mean()) / (df.max() - df.min())
df_train_cont, df_test_cont = df[:149], df[149:]
df_train = df_train_cont[df_train_cont.columns[1:10]]
df_labels = df_train_cont[df_train_cont.columns[10]]
df_test = df_test_cont[df_test_cont.columns[1:10]]
df_test_label = df_test_cont[df_test_cont.columns[10]]
trainList = df_train.values.tolist()
trainLabels = df_labels.values.tolist()
testList = df_test.values.tolist()
testLabel = df_test_label.values.tolist()
testLabelnp = np.array(testLabel)


clf = svm.SVC()
clf.fit(trainList, trainLabels)


predict = clf.predict(testList)

sub = np.subtract(testLabelnp, predict)

errorCount = 0
i = 0
for i in range(len(sub)):
    if sub[i]!=0:
        errorCount+=1
    else:
        pass

errorPct = (errorCount*100.0)/len(sub)
accuracy = 100-errorPct

print "accuracy is %s" %errorPct + "%"


import matplotlib.pyplot as plt
%matplotlib inline
from pandas.tools.plotting import scatter_matrix
#df_train_plotX = df_train_cont[df_train_cont.columns[1]]
#df_train_plotY = df_train_cont[df_train_cont.columns[3]]
df_plot = df_norm[df_norm.columns[1:10]]
scatter_matrix(df_plot, alpha=0.2, figsize=(20, 20), diagonal='kde')



