# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 08:22:25 2025

@author: Hp
"""

import pandas as pd
import numpy as np
letters=pd.read_csv("letterdata.csv")
letters.head()
letters.describe()
'''
datase typically used for handwritten character recognition
or related  machine learning tasks,Here is a breakdown ot its structure

letter:represents the target class (the letter being identified
features (xbox to yedgex):these are numeric attributes
descibing various geometric or statistical properties of the character
xbox and ybox:X and Y bounding box dimensions
width and height :width and height of the character bounding box
onpix:number of on pixel in the character image
xbar and ybar: mean X and Y coordibate values of on pixels
x2bar,y2bar and xybar: variamce and covariance of pixel intensities
x2bar,and xy2bar additioanl stastical metrics
for special                                    '''

#let us try carry out EDA
a=letters.describe()
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
train,test=train_test_split(letters,test_size=0.2)

#let us split the data in X and y for both trai and test data
train_X=train.iloc[:,1:]
train_y=train.iloc[:,0]
test_X=train.iloc[:,1:]
test_y=train.iloc[:,0]

#kernal linear
model_linear=SVC(kernel='linear')
model_linear.fit(train_X,train_y)
pred_test_linear=model_linear.predict(test_X)

#Now let us check the accuracy
np.mean(pred_test_linear==test_y)
# 0.874

#kernel rbf
model_rbf=SVC(kernel='rbf')
model_rbf.fit(train_X,train_y)
pred_test_rbf=model_rbf.predict(test_X)

#Now let us check the accuracy
np.mean(pred_test_rbf==test_y)
# 0.9366875
































