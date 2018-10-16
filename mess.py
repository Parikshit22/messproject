# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 12:17:01 2018

@author: MUJ
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics




dataset = pd.read_csv('messdataset.csv')


print(dataset.shape)
print(dataset.describe())
dataset.hist(bins= 10,figsize =(5,5))
plt.show()

train_set,test_set = train_test_split(dataset, test_size = 0.2 , random_state = 42)
print(len(train_set),"train+",len(test_set),"test")
print(train_set)
test_set_label = test_set["footfall"].copy()
test_set_att = test_set.drop("footfall",axis=1)
dataset = train_set.copy()


dataset1 = dataset
dataset_label = dataset["footfall"].copy()
dataset = dataset.drop("footfall",axis=1)

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
dataset=MultiColumnLabelEncoder(columns = ['day','week','weather','holiday','Special','meal type']).fit_transform(dataset)
test_set_att=MultiColumnLabelEncoder(columns = ['day','week','weather','holiday','Special','meal type']).fit_transform(test_set_att)

scaler = StandardScaler()
print(scaler.fit(dataset))
dataset = scaler.transform(dataset)
print(dataset)
test_set_att = scaler.transform(test_set_att)

lin_reg = LinearRegression()
lin_reg.fit(dataset,dataset_label)
dataset_prediction=lin_reg.predict(test_set_att)
lin_mse = mean_squared_error(test_set_label,dataset_prediction)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)


accuracy = lin_reg.score(test_set_label, dataset_prediction)
print(accuracy*100,'%')
