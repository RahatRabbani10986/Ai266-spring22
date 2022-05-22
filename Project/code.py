"Linear Regression"
#Date:May 21,2022
#---Importing Libraries---#

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import numpy as np

#-----This line is reading train Data---#
trainDf = pd.read_csv("E:/ERROR/train.csv")

#-----This line is reading test Data---#
testDf = pd.read_csv('E:/ERROR/test.csv')

#-----This line is printing train Data---#
print("Sample Train data: ")
print(trainDf.head())

#-----This line is printing test Data---#
print("Sample test data: ")
print(testDf.head())

#-----Cleaning Data---#
del trainDf['id']
del trainDf['f_27']
del testDf['f_27']

#-------Drop the train data target value and then it to other----#
train = trainDf.drop(columns=['target'])
new_train = trainDf['target']
o_train, o_test, n_train, n_test = train_test_split(train, new_train, test_size=0.2, random_state=35)

#----Applying Linear Regression------#
#---Importing Library---#
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(o_train, n_train)
predict = LR.predict(o_test)
acc_LR = round(LR.score(o_train, n_train) * 100, 2)

#Cross Validation
#---Importing Library---#
from sklearn.model_selection import cross_val_score
acc_LR
clf = LinearRegression()
scores = cross_val_score(clf, o_train, n_train, cv=8)
mean = scores.mean()
print('Linear Regression Accuracy after CV: ',scores)
print('Linear Regression Accuracy after CV: ',mean)

#---For any duplicate Data cleaning---#
"""dups = trainDf.duplicated()
# report if there are any duplicates
print(dups.any())
# All duplicate rows
print(trainDf[dups])
print(trainDf.shape)
# delete duplicate rows
trainDf.drop_duplicates(inplace=True)
print(trainDf.shape)"""

trainDff = testDf[['id']]
trainDff
train_predict = testDf.drop(columns=['id'])
train_predict.head()
Test_predict = LR.predict(train_predict)
trainDff['target'] = Test_predict
trainDff.head()
print(trainDff)
trainDff.to_csv('LinearRegression.csv', index=False)

#KFold
#---Importing Library---#
from sklearn.model_selection import KFold
kf = KFold(n_splits=2)
for trainDf, testDf in kf.split(train):
    print("TRAIN:\n %s \nTEST:\n %s" % (trainDf, testDf))
    
    
    
#"Logistic Regression"
#Date:May 21,2022
#---Importing Libraries---#

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

#-----This line is reading train Data---#
trainDf = pd.read_csv("E:/ERROR/train.csv")

#-----This line is reading test Data---#
testDf = pd.read_csv('E:/ERROR/test.csv')

#-----This line is printing train Data---#
print("Sample Train data: ")
print(trainDf.head())

#-----This line is printing test Data---#
print("Sample test data: ")
print(testDf.head())

#-----Cleaning Data---#
del trainDf['id']
del trainDf['f_27']
del testDf['f_27']

#-------Drop the train data target value and then it to other----#
train = trainDf.drop(columns=['target'])
new_train = trainDf['target']
o_train, o_test, n_train, n_test = train_test_split(train, new_train, test_size=0.2, random_state=35)



#----Applying Logistic Regression------#
#---Importing Library---#
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(o_train, n_train)
predict = LR.predict(o_test)
acc_LR = round(LR.score(o_train, n_train) * 100, 2)

print("Accuracy")
#Cross Validation
#---Importing Library---#
from sklearn.model_selection import cross_val_score
acc_LR
clf = LogisticRegression()
scores = cross_val_score(clf, o_train, n_train, cv=8)
mean = scores.mean()
print('Logistic Regression Accuracy after CV: ',scores)
print('Logistic Regression Accuracy after CV: ',mean)

#---For any duplicate Data cleaning---#
"""dups = trainDf.duplicated()
# report if there are any duplicates
print(dups.any())
# All duplicate rows
print(trainDf[dups])
print(trainDf.shape)
# delete duplicate rows
trainDf.drop_duplicates(inplace=True)
print(trainDf.shape)"""

trainDff = testDf[['id']]
trainDff
train_predict = testDf.drop(columns=['id'])
train_predict.head()
Test_predict = LR.predict(train_predict)
trainDff['target'] = Test_predict
trainDff.head()
print(trainDff)
trainDff.to_csv('LogisticRegression.csv', index=False)



#KFold
from sklearn.model_selection import KFold
kf = KFold(n_splits=2)
for trainDf, testDf in kf.split(train):
    print("TRAIN:\n %s \nTEST:\n %s" % (trainDf, testDf))

    
"SVM"
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

train = pd.read_csv('train.csv')
print(train.head(5))

test = pd.read_csv('test.csv')
test.head(2)

train.isnull().sum()
train['target'] = train['target'].map({'Class_1':1,'Class_2':2,'Class_3':3,'Class_4':4})
print(train.head(2))
train.drop(['id'],axis=1, inplace=True)
test.drop(['id'],axis=1, inplace=True)

X = train.drop(['target'], axis=1)
y = train['target']

print(X.head(2))

print(y.value_counts())


from sklearn import svm
from sklearn.model_selection import cross_val_score

clf = svm.SVC(kernel='linear', C=1, random_state=42)
clf.fit(X, y)

print(clf.score(X, y))
scores = cross_val_score(clf, X, y, cv=5)

print(scores)

#knn
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):        
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]        
        k_idx = np.argsort(distances)[: self.k]        
        k_neighbor_labels = [self.y_train[i] for i in k_idx]        
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]


if __name__ == "__main__":
    
    from matplotlib.colors import ListedColormap
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    

plt.plot(X_train, label='Testing dataset Accuracy')

plt.plot(y_train, label='Training dataset Accuracy')

plt.legend()

plt.xlabel('X-AXIX')

plt.ylabel('Y-AXIX')
plt.show()
