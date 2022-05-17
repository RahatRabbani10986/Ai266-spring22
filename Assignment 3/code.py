# Rahat Rabani 
# Huzaifa Shakeel
# Salah Shakeel
# Aiman Akmal
# Ahmed Khan
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

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
