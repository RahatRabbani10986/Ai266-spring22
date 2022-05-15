#Date:May 11,2022
#---Importing Libraries---#
import pandas as pd
import numpy as np


#-----This line is reading test Data---#
testDf = pd.read_csv('sample.csv')

#-----In this peice of code we are getting ID and removing everything else----#
idDf = testDf[['id']];

#----This line is creating target column as per kaggle----#
idDf.insert(1,"target",0)

#----This line is generating random values-----#
idDf['target']=np.random.rand(700000,1);

#----Writing Back in Csv File----#
idDf.to_csv('sample_file.csv');

print(idDf);
