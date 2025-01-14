# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 09:46:28 2025

@author: Hp
"""
##Multiple correlation regression analysis
import pandas as pd
import numpy as np
cars=pd.read_csv("Cars.csv")
#Exploratory data analysis
#1.Measure the central tendency
#2.Measure the dispersion
#3.Third moment business decision
#4.Fourth moment business decision
#5.probability distribution
#6.Graphical representation(Histogram,Boxplot)
cars.describe()

#Graphical representation
import matplotlib.pyplot as plt
plt.bar(height=cars.HP,x=np.arange(1,82,1))
plt.hist(cars.HP)
plt.boxplot(cars.HP)
#There are several outliers in HP columns
#similar operations are expected for other three columns

#Now let us plot joint plot,it is to show scatter plot and histogram
import seaborn as sns
sns.jointplot(x=cars['HP'],y=cars['MPG'])

#Now let us plot count plot
plt.figure(1,figsize=(16,10))
sns.countplot(cars['HP'])
#count plot shows how many times the each value occured
#92 HP value accured 7 times

#QQplot
from scipy import stats
import pylab
stats.probplot(cars.MPG,dist = "norm", plot=pylab)
plt.show()
# MPG data is normally distributed
# There are 10 scatter plot need to be plotted , one by one
# in sequence to plot, so we can use pair plots

import seaborn as sns
sns.pairplot(cars.iloc[:,:])
#you can check the collinearity problem between the input variable
#you can check plot between SP and HP they are strongly corelated
#same way you can check WT and VOL,it is also strongly corelated

#Now let us check r value between variables
cars.corr()
#you can check SP and HP r value is 0.97
#same way WT and VOL it has got 0.999 which is greater

#NOw although we observed strongly corelated pairs
#still we will go for linear regression
import statsmodels.formula.api as smf
ml1=smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()
ml1.summary()
#r square value observed is 0.771<0.85
#p value of WT and VOL is 0.814 and 0.556 which is very high
#it means it is gfeater than 0.05,WT and VOl col needs to be ignored
#or delete.
#Instead od deleting all entries let us check row wise outliers
#indentify if there is any influenyial value
#to check you can yse influential index
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
#76 is the value which has got outliers
#go to the dataframe and check 76th entry
#Let us delete that entry
cars_new=cars.drop(cars.index[[76]])

#again apply regression  to cars_new
ml1_new=smf.ols('MPG~WT+VOL+SP+HP',data=cars_new).fit()
ml1_new.summary()
#R-square value is 0.819 but p value are same hence not solving the purpose
#Now next option is delete the column but
#question is which option is to be deleted
#we have already checked correlation factor r
#VOL has got -0.529 and for WT =-0.526
#WT is less hence can be deleted

#Another approach is to check the collonearity,rsquare is given
#that value we will have to apply regression w.r.t x1 and input
#as x2,x3 and x4 so on so forth
rsq_hp=smf.ols('MPG~WT+VOL+SP',data=cars).fit().rsquared
vif_hp=1/(1-rsq_hp)
#VIF is variance influential factor calculating VIF helps to
#of x1 w.r.t x2,x3 and x4

rsq_wt=smf.ols('MPG~HP+VOL+SP',data=cars).fit().rsquared
vif_wt=1/(1-rsq_wt)

rsq_vol=smf.ols('MPG~HP+WT+SP',data=cars).fit().rsquared
vif_vol=1/(1-rsq_vol)

rsq_sp=smf.ols('MPG~HP+WT+VOL',data=cars).fit().rsquared
vif_sp=1/(1-rsq_sp)

#vif_wt=639.53,vif_vol=638.80 hence vif_wt is grater,thumb rule is vif sholud not be greater than 10

#storing the values in dataframe
d1={'Variables':['HP','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
vif_frame=pd.DataFrame(d1)
vif_frame

#let us drop WT and apply correlation to remaining three
final_ml=smf.ols('MPG~HP+VOL+SP',data=cars).fit()
final_ml.summary()
#r square is  0.770 and p value 0.00,0.012<0.05

#prediction
pred=final_ml.predict(cars)

#QQ plot
res=final_ml.resid
sm.qqplot(res)
plt.show()
#This qq plot is on residual which os obtained on training data
#errors are obtained on test data
stats.probplot(res,dist='norm',plot=pylab)
plt.show()

#let us plot the residual plot which takes the residual values and the data
sns.residplot(x=pred,y=cars.MPG,lowess=True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
#residual plots are used to check whether the error are indepemdent or not

#let us plot the influentia plot
sm.graphics.influence_plot(final_ml)
#we have taken cars instead cars_newdata,hence 76 is reflected again in influential plot

#splitting the data into train and test data
from sklearn.model_selection import train_test_split
cars_train,cars_test=train_test_split(cars,test_size=0.2)
#preparing the model on train data
model_train=smf.ols('MPG~VOL+SP+HP',data=cars_train).fit()
model_train.summary()
test_pred=model_train.predict(cars_test)
#Test errors
test_error=test_pred-cars_test.MPG
test_rmse=np.sqrt(np.mean(test_error*test_error))
test_rmse



















