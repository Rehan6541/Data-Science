# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 09:17:10 2025

@author: Hp
"""


import pandas as pd
walmart=pd.read_csv("Walmart_Footfalls_Raw.csv")
walmart.dtypes
month=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec',]
#In walmart data we have Jan-1991 in oth column ,we need only first
#Example-Jan from each cell
p=walmart["Month"][0]
p[0:3]

#Before we will extract ,let us create new column called months to store extracted values
walmart['month']=0
#you can check the dataframe with monthhs name with all values 0
#the total records are 159 in walmmart
for i in range(159):
    p=walmart["Month"][i]
    walmart["month"][i]=p[0:3]
#for all these months create dummy variables
month_dummies=pd.DataFrame(pd.get_dummies(walmart['month']))
## now let us concatenate these dummy values to dataframe
walmart1=pd.concat([walmart,month_dummies],axis=1)
# you can check the dataframe walmart1

# similarly we need to create column 't'
import numpy as np
walmart1['t']=np.arange(1,160)
walmart1['t_squared']=walmart1['t']*walmart1['t']
walmart1['log_footfalls']=np.log(walmart1['Footfalls'])
walmart1.columns

#now lets us check the visuals of the footfalls
walmart1.Footfalls.plot()
#You will get  trend with gradual  increasing and linear

#We have to forecast Sales in next 1 year,hence horizon=12,even 
#season=12,so validating data will be 12 and training will 159-12=147
Train=walmart1.head(147)
Test=walmart1.tail(12)

# Now let us apply linear regression
import statsmodels.formula.api as smf

##Linear model
linear_model=smf.ols("Footfalls~t",data=Train).fit()
pred_linear=pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear=np.sqrt(np.mean((np.array(Test['Footfalls'])-np.array(pred_linear))**2))
rmse_linear

##Exponential model
Exp_model=smf.ols("log_footfalls~t",data=Train).fit()
pred_Exp=pd.Series(Exp_model.predict(pd.DataFrame(Test['t'])))
rmse_Exp=np.sqrt(np.mean((np.array(Test['Footfalls'])-np.exp(pred_Exp))**2))
rmse_Exp

##Quadratic model
Quad=smf.ols("Footfalls~t+t_squared",data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t", "t_squared"]]))
rmse_Quad=np.sqrt(np.mean((np.array(Test['Footfalls'])-np.array(pred_Quad))**2))
rmse_Quad

################### Additive seasonality ########################
add_sea = smf.ols('Footfalls~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
add_sea.summary()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_add_sea))**2))
rmse_add_sea

##Multiplicative seasonability model
mul_sea=smf.ols("log_footfalls~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov",data=Train).fit()
mul_sea.summary()
pred_mul_sea = pd.Series(mul_sea.predict(Test))
rmse_mul_sea=np.sqrt(np.mean((np.array(Test['Footfalls'])-np.array(np.exp(pred_mul_sea)))**2))
rmse_mul_sea

################### Additive seasonality with quadratic trend ########################
add_sea_quad = smf.ols('Footfalls~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov', data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad

##Multiplicative seasonability linear model
mul_add_sea=smf.ols("log_footfalls~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov",data=Train).fit()
pred_mul_add_sea = pd.Series(mul_add_sea.predict(Test))
rmse_mul_add_sea=np.sqrt(np.mean((np.array(Test['Footfalls'])-np.array(np.exp(pred_mul_add_sea)))**2))
rmse_mul_add_sea

###let us create a dataframe and add all these rmse_values
data={"Model":pd.Series(['rmse_linear','rmse_Exp','rmse_Quad','rmse_add_sea','rmse_mul_sea','rmse_add_sea_quad','rmse_mul_add_sea']),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_mul_sea,rmse_add_sea_quad,rmse_mul_add_sea])}
data
#line 82 -Additive seasonality with quadratic trend  is lowest
#Additive seasonality with quadratic trend has got lowest value and accuracy better

## Now let us test the model with full data
predict_data = pd.read_excel("Predict_new.xlsx")
model_full=smf.ols("Footfalls~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov",data=walmart1).fit()
pred_new = pd.Series(model_full.predict(predict_data))
pred_new
predict_data["forecasted_footfalls"] = pd.Series(pred_new)
predict_data
#You check predict_data dataframe
























