import numpy as np
csv = np.genfromtxt ('datasets/car/glass.csv', delimiter=",")
np.random.shuffle(csv)
# print csv[150]
training=csv[0:150]
# print training
X=training[:,1:10]
# print X[0]
# print training[0,10]
Y=training[:,10:11]
# print Y
test=csv[150:214]
X_test=test[:,1:10]
# Y_test=test[:,10:11]
Y_test=test[:,10]
from sklearn.svm import SVC
clf=SVC(C=10,kernel='rbf')
clf.fit(X,Y)
predict=clf.predict(X_test)
sub=np.subtract(predict,Y_test)
print sub
predictArr=np.array(predict)
# print predict
# print Y_test
# print predictArr
count=0
for p in sub:
    if(p!=0):
        count=count+1
print count