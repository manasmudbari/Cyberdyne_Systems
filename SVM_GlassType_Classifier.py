
# coding: utf-8

# In[109]:

import pandas as pd
import os
import numpy as np

df = pd.read_csv("~/Desktop/glassType.txt")

df = df.iloc[np.random.permutation(len(df))]


# In[110]:

df_train_cont, df_test_cont = df[:149], df[150:]

df_train = df_train_cont[df_train_cont.columns[1:10]]
df_labels = df_train_cont[df_train_cont.columns[10]]
df_test = df_test_cont[df_test_cont.columns[1:10]]
df_test_label = df_test_cont[df_test_cont.columns[10]]
from sklearn import svm
trainList = df_train.values.tolist()
trainLabels = df_labels.values.tolist()
clf = svm.SVC()
clf.fit(trainList, trainLabels)


# In[111]:

testList = df_test.values.tolist()
testLabel = df_test_label.values.tolist()
testLabelnp = np.array(testLabel)
predict = clf.predict(testList)

sub = np.subtract(testLabelnp, predict)


# In[125]:
#counting error
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



