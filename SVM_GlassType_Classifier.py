
# coding: utf-8

import pandas as pd
import os
import numpy as np
from sklearn import svm

def readData(cutoff, id1, idn):
	df = pd.read_csv("~/Desktop/glassType.txt")
	df = df.iloc[np.random.permutation(len(df))]
	df_train_cont, df_test_cont = df[:cutoff], df[cutoff+1:]
	df_train = df_train_cont[df_train_cont.columns[id1:idn]]
	df_labels = df_train_cont[df_train_cont.columns[idn]]
	df_test = df_test_cont[df_test_cont.columns[id1:idn]
	df_test_label = df_test_cont[df_test_cont.columns[idn]]
	trainList = df_train.values.tolist()
	trainLabels = df_labels.values.tolist()
	testList = df_test.values.tolist()
	testLabel = df_test_label.values.tolist()
	testLabelnp = np.array(testLabel)
	return trainList, trainLabels, testList, testLabelnp

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


# In[ ]:



