import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("~/Desktop/adult.txt", header=None)

df = df.iloc[np.random.permutation(len(df))]

df.columns = ["age", "workclass", "fnlwgt", "education", "education-num", "maritalstatus", "occupation", "relationship", 
              "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "nativecountry", "salary"]


#Processing each column with categorical value
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

df.workclass = le.fit_transform(df.workclass)
df.education = le.fit_transform(df.education)
df.maritalstatus = le.fit_transform(df.maritalstatus)
df.occupation = le.fit_transform(df.occupation)
df.relationship = le.fit_transform(df.relationship)
df.race = le.fit_transform(df.race)
df.sex = le.fit_transform(df.sex)
df.nativecountry = le.fit_transform(df.nativecountry)
df.salary = le.fit_transform(df.salary)

df_train_cont, df_test_cont = df[:22793], df[22793:]

df_train = df_train_cont[df_train_cont.columns[0:14]]
df_labels = df_train_cont[df_train_cont.columns[14]]
df_test = df_test_cont[df_test_cont.columns[0:14]]
df_test_label = df_test_cont[df_test_cont.columns[14]]

train_matrix = df_train.as_matrix()

trainList = df_train.values.tolist()
trainLabels = df_labels.values.tolist()