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
model3=smf.ols("Waist~np.log(AT)",data=wcat).fit()
#Y is np.log(AT) and X is waist
model3.summary()
#R-squarred=0.707<0.85,there is scope of improvement
#P=0.0<0.05 hence accepatable
#bita~0=0.7410
#bita~1=0.0403
pred3=model3.predict(pd.DataFrame(wcat.AT))
pred3

#Regression line
plt.scatter(wcat.Waist,np.log(wcat.AT))
plt.plot(np.log(wcat.AT),pred3,'r')
plt.legend(['Predicted Line','Observed data'])
plt.show


#Error calculation
res3=wcat.AT-pred3
np.mean(res3)
res_sqr3=res3*res3
msel=np.mean(res_sqr3)
rmsel=np.sqrt(msel)
rmsel
#32.49688490932126









