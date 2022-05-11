#Date:May 11,2022
#---Importing Libraries---#
import pandas as pd
import numpy as np

#-----Reading test Data---#
testDf = pd.read_csv('sample.csv')

#-----Getting Id and removing everything else----#
idDf = testDf[['id']];

#----Creating target column as per kaggle----#
idDf.insert(1,"target",0)

#----Generating Random Values-----#
idDf['target']=np.random.rand(700000,1);

#----Writing Back in Csv File----#
idDf.to_csv('sample_file.csv');
print(idDf);
