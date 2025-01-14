# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 10:13:27 2025

@author: Hp
"""

import pandas as pd
import numpy as np
import seaborn as sns
wcat=pd.read_csv("wc-at.csv")
#EDA
wcat.info
wcat.describe()
#Average waist is 91.90 and min is 63.50 and max is 121
#Average AT is 101.89 and min is 11.44 and max is 253
import matplotlib.pyplot as plt
plt.bar(height=wcat.AT,x=np.arange(1,110,1))
sns.displot(wcat.AT)
#Data is normal but right skewed
plt.boxplot(wcat.AT)
#No outliers but right skewed
plt.bar(height=wcat.Waist,x=np.arange(1,110,1))
sns.displot(wcat.Waist)
#Data is normal bimodal
plt.boxplot(wcat.Waist)
#No outliers but right skewed


#Bivariant analysis
plt.scatter(x=wcat.Waist,y=wcat.AT)
#Data is linearly scattered,direction positive,strength:poor
#Now let us check the correlation coeficient
np.corrcoef(wcat.Waist,wcat.AT)
#The corellation coeficient is 0.81855781.85 hence the correlation
#Let us check the direction of correlation
cov_output=np.cov(wcat.Waist,wcat.AT)[0,1]
cov_output
#635.9100064135235 is positive means correlation will be positive


#let us apply to various models and check the feasibilty
import statsmodels.formula.api as smf
#First simple linear model
model=smf.ols("AT~Waist",data=wcat).fit()
#Y is AT and X is waist
model.summary()
#R-squarred=0.670<0.85,there is scope of improvement
#P=0.0<0.05 hence accepatable
#bita~0=-215.9815
#bita~1=3.4589
pred1=model.predict(pd.DataFrame(wcat.Waist))
pred1
#Regression line
plt.scatter(wcat.Waist,wcat.AT)
plt.plot(wcat.Waist,pred1,'r')
plt.legend(['Predicted Line','Observed data'])
plt.show


#Error calculation
res1=wcat.AT-pred1
np.mean(res1)
res_sqr1=res1*res1
msel=np.mean(res_sqr1)
rmsel=np.sqrt(msel)
rmsel
#32.760177495755144

***************************************************************************
#Let us try another model
#X=log(Waist)
plt.scatter(x=np.log(wcat.Waist),y=wcat.AT)
#Data is linearly scattered,direction positive,strength:poor
#Now let us check the correlation coeficient
np.corrcoef(np.log(wcat.Waist),wcat.AT)
#The corellation coeficient is 0.82177819<0.85 hence the correlation
#r=0.8217
model2=smf.ols("AT~np.log(Waist)",data=wcat).fit()
#Y is AT and X is log(waist)
model2.summary()
#R-squarred=0.675<0.85,there is scope of improvement
#P=0.0<0.05 hence accepatable
#bita~0=-1328.3420
#bita~1=317.1356
pred2=model2.predict(pd.DataFrame(wcat.Waist))
pred2

#Regression line
plt.scatter(np.log(wcat.Waist),wcat.AT)
plt.plot(np.log(wcat.Waist),pred2,'r')
plt.legend(['Predicted Line','Observed data'])
plt.show


#Error calculation
res2=wcat.AT-pred2
np.mean(res2)
res_sqr2=res2*res2
msel=np.mean(res_sqr2)
rmsel=np.sqrt(msel)
rmsel
#32.49688490932126


*********************************************************
#Let us try another model
#Y=log(AT)
plt.scatter(x=wcat.Waist,y=np.log(wcat.AT))
#Data is linearly scattered,direction positive,strength:poor
#Now let us check the correlation coeficient
np.corrcoef(wcat.Waist,np.log(wcat.AT))
#The corellation coeficient is 0.84090069<0.85 hence the correlation
#r=0.8409
model3=smf.ols("np.log(AT)~Waist",data=wcat).fit()
#Y is np.log(AT) and X is waist
model3.summary()
#R-squarred=0.707<0.85,there is scope of improvement
#P=0.0<0.05 hence accepatable
#bita~0=0.7410
#bita~1=0.0403
pred3=model3.predict(pd.DataFrame(wcat.Waist))
pred3_at=np.exp(pred3)
pred3_at

#Regression line
plt.scatter(wcat.Waist,np.log(wcat.AT))
plt.plot(wcat.Waist,pred3,'r')
plt.legend(['Predicted Line','Observed data_model3'])
plt.show


#Error calculation
res3=wcat.AT-pred3_at
res_sqr3=res3*res3
msel=np.mean(res_sqr3)
rmsel3=np.sqrt(msel)
rmsel3
#38.52900175807143
#There are no significant change r=0.8409,r^2=0.707 and rsme=38.52900175807143


*************************************************************************
#Hence Let us try another model
#Y=log(AT) and X=Waist,X*X=Waist.Waist
#Polynomial model
#Here r can not be calculated
model4=smf.ols("np.log(AT)~Waist+I(Waist*Waist)",data=wcat).fit()
#Y is np.log(AT) and X is waist
model4.summary()
#R-squarred=0.779<0.85,there is scope of improvement
#P=0.0<0.05 hence accepatable
#bita~0=-7.8241
#bita~1=0.2289 
pred4=model4.predict(pd.DataFrame(wcat.Waist))
pred4
pred4_at=np.exp(pred4)
pred4_at

#Regression line
plt.scatter(wcat.Waist,np.log(wcat.AT))
plt.plot(wcat.Waist,pred4,'r')
plt.legend(['Predicted Line','Observed data_model4'])
plt.show


#Error calculation
res4=wcat.AT-pred4_at
res_sqr4=res4*res4
msel=np.mean(res_sqr4)
rmsel=np.sqrt(msel)
rmsel
#32.244447827762464


#We have to generalize the best model
from sklearn.model_selection import train_test_split
train,test=train_test_split(wcat,test_size=0.2)
plt.scatter(train.Waist, np.log(train.AT))
plt.scatter(test.Waist, np.log(test.AT))
final_model=smf.ols('np.log(AT)~Waist+I(Waist*Waist)',data=wcat).fit()
#Y is np.log(AT) and X is waist
final_model.summary()
#R-squarred=0.779<0.85,there is scope of improvement
#P=0.0<0.05 hence accepatable
#bita~0=-7.8241
#bita~1=0.2289 
test_pred=final_model.predict(pd.DataFrame(test))
test_pred_at=np.exp(test_pred)
test_pred_at

train_pred = final_model.predict(pd.DataFrame(train))
train_pred_at = np.exp(train_pred)
train_pred_at

#Evalution on test data
test_res=test.AT-test_pred_at
test_sqr=test_res*test_res
test_mse=np.mean(test_sqr)
test_rmse=np.sqrt(test_mse)
test_rmse
#31.166396245162282

#Evalution on train data
train_res=train.AT-train_pred_at
train_sqr=train_res*train_res
train_mse=np.mean(train_sqr)
train_rmse=np.sqrt(train_mse)
train_rmse
#32.51139635824184

#test_rmse>train_rmse















