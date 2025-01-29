# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:03:27 2025

@author: Hp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import classification_report
claimant=pd.read_csv("claimants.csv")
#The 0th column is CASENUM which is not useful,hence drop it
c1=claimant.drop("CASENUM",axis=1)
c1.head()
c1.describe()

#let us check the null value
c1.isna().sum()
#There are several null values around 290
#continuous data null use mean imputation
#categorical data null use mode imputation
mean_value=c1.CLMAGE.mean()
mean_value
c1.CLMAGE=c1.CLMAGE.fillna(mean_value)
c1.CLMAGE.isna().sum()

#for decrete value like CLMSEX we need to use mode imputation
mode_CLMSEX=c1.CLMSEX.mode()
mode_CLMSEX
#here if you will observe the output it is 0 1 i.e
#mode_CLMSEX[0]=0,mode_CLMSEX[1]=1,we are passing mode_CLMSEX[0]
c1.CLMSEX=c1.CLMSEX.fillna((mode_CLMSEX)[0])
c1.CLMSEX.isna().sum()

#CLMINSUR
mode_INSUR=c1['CLMINSUR'].mode()
mode_INSUR
c1.CLMINSUR=c1.CLMINSUR.fillna((mode_INSUR)[0])
c1.CLMINSUR.isna().sum()

#Seat belt
mode_SEATBELT=c1["SEATBELT"].mode()
mode_SEATBELT
c1.SEATBELT=c1.SEATBELT.fillna((mode_SEATBELT)[0])
c1.SEATBELT.isna().sum()

##MOdel Building
logit_model=sm.logit('ATTORNEY~CLMAGE+LOSS+CLMINSUR+SEATBELT',data=c1).fit()
logit_model.summary()
logit_model.summary2()
#let us go for prediction
pred=logit_model.predict(c1.iloc[:,1:])
#####################################
#To derive ROC curve
#ROC curve has tpr on y axis and fpr on x axis ideally tpr must br high
#fpr must be low
fpr,tpr,thresholds=roc_curve(c1.ATTORNEY,pred)
#to identify optimum threshold
optimal_idx=np.argmax(tpr-fpr)
optimal_threshold=thresholds[optimal_idx]
optimal_threshold
#0.5712938792809786, by default you can take 0.5 value as a threshold
#now we want to identify if a new value is given to the model it will
#fail in  which region 0 or 1,for that we need to derive ROC curve
#to draw ROC curve
import pylab as pl
i=np.arange(len(tpr))
roc=pd.DataFrame({
    'fpr': pd.Series(fpr,index=i),
    'tpr': pd.Series(tpr,index=i),
    '1-fpr': pd.Series(1-fpr,index=i),
    'tf': pd.Series(tpr-(1-fpr),index=i),
    'threshold': pd.Series(thresholds,index=i)
    })

#This code creates a dataframe called roc using pandas (pd).
#It organizes various metrics related to the REceiver Operating Characteristic
#into columns.Each column represents a specific metric and the rows are index
#plot ROC curve
plt.plot(fpr,tpr)
plt.xlabel("False positive rate");plt.ylabel('True positive rate')
roc_auc=auc(fpr,tpr)
print("Area under the curve %f"%roc_auc)

#Now let us add prediction coulmn in dataframe
c1['pred']=np.zeros(1340)
c1.loc[pred>optimal_threshold,"pred"]=1
#if predicted value is greater than optimal threshold then change pred column a
#Classification report
classification=classification_report(c1["pred"],c1["ATTORNEY"])
classification

#spliting the data into train and test data
from sklearn.model_selection import train_test_split
train_data,test_data=train_test_split(c1,test_size=0.3)
#model building using
model=sm.logit('ATTORNEY~CLMAGE+LOSS+CLMINSUR+SEATBELT',data=train_data).fit()
model.summary()
model.summary2()
#AIC is 1137.9338

#prediction on test data
test_pred=model.predict(test_data)
test_data["test_pred"]=np.zeros(402)

#taking threshold value as optimal threshold value
test_data.loc[test_pred>optimal_threshold,'test_pred']=1

#confusion matrix
confusion_matrix=pd.crosstab(test_data.test_pred, test_data.ATTORNEY)
confusion_matrix
accuracy=(148+142)/402
accuracy

#classification report
classification=classification_report(test_data["test_pred"],test_data["ATTORNEY"])
classification

#ROC curve and AUC
fpr,tpr,thresholds=metrics.roc_curve(test_data.ATTORNEY,test_pred)
#plot of ROC 
plt.plot(fpr,tpr);plt.xlabel("False positive rate");plt.ylabel('True positive rate')
roc_auc_test=metrics.auc(fpr,tpr)
print("Area under the curve %f"%roc_auc_test)

#predicting on train data
train_pred=model.predict(train_data.iloc[:,1:])

#creating new column
train_data["train_pred"]=np.zeros(938)
train_data.loc[train_pred>optimal_threshold,"train_pred"]=1

#confusion matrix
confusion_matrix=pd.crosstab(train_data.train_pred, train_data.ATTORNEY)
confusion_matrix
accuracy=(326+334)/938
accuracy

#classification report
classification_train=classification_report(train_data["train_pred"],train_data["ATTORNEY"])
classification_train

#ROC curve and AUC
fpr,tpr,thresholds=metrics.roc_curve(train_data.ATTORNEY,train_pred)
#plot of ROC 
plt.plot(fpr,tpr);plt.xlabel("False positive rate");plt.ylabel('True positive rate')
roc_auc_train=metrics.auc(fpr,tpr)
print("Area under the curve %f"%roc_auc_train)

































