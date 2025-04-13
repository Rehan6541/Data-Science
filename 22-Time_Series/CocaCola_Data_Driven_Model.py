# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 09:07:30 2025

@author: Hp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # Holt Winter's Exponential Smoothing

#Now load the dataset
cocacola= pd.read_excel("CocaCola_Sales_Rawdata.xlsx")

#Let us plot the dataset and its nature
cocacola.Sales.plot() # time series plot 
#Linearly increasing slight trend and season

#Splitting the data into Train and Test data
#Since we are working on the quartely datasets and in year there are 4 quarters
#Train dataset=38
#Recent 4 time period values are Test data,
Train = cocacola.head(38)
Test = cocacola.tail(4)

#Now we are considering performance parameters as mean absolute
#Percentage error
#Rather than mean square value
#Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)


#EDA which comprises indentification of level,trends and seasonability
#In order to separate trend and sesonability moving average can be done
mv_pred = cocacola["Sales"].rolling(4).mean()
mv_pred.tail(4)
#This calculates a 4-period Moving average for the Sales column

#If smooths the data by taking the mean of the last 4 quarters at each point,
#Now let us calculate mean absolute percentage of these
#Values
#Moving Average for the time series
MAPE(mv_pred.tail(4), Test.Sales)
#Moving average helps to extract trebd and seasonability
#MAPE evaluates prediction accuracy
#If MAPE is low,the model is performing well
#moving average is predicting complete value,out of which last 4
#are considered as predicted values and last four values of Test.Sales
#Basic pupose of moving average is desasonability

#Plot with Moving Averages
cocacola.Sales.plot(label = "org")
#This is original plot

#Now let us separate out trend ans Seasonability
for i in range(2, 9, 2):
    #It will take windoq size 2,4,6,8
    cocacola["Sales"].rolling(i).mean().plot(label = str(i))
plt.legend(loc = 3)
#You can see i=4 and 8 are deseasonable plotss


#Time series decomposition is the process of separating data into its core components.
#Time series decomposition plot using Moving Average
decompose_ts_add = seasonal_decompose(cocacola.Sales, model = "additive", period = 4)
print(decompose_ts_add.trend)
print(decompose_ts_add.seasonal)
print(decompose_ts_add.resid)
print(decompose_ts_add.observed)
decompose_ts_add.plot()
'''
trend component 

There is a visible upward trend in sales over time
indicatiing overall growth
Seasonal Component
 A clear repeating pattern is visible every four quarters (one year),
 confiming striong sesonality in sales
 Residual component
 
 The residual \s (random fluctuations) should ideally n=be white noise
 if they show patterns , it suggests that other factors might be influencing  sales
'''

decompose_ts_mul = seasonal_decompose(cocacola.Sales, model = "multiplicative", period = 7)
print(decompose_ts_mul.trend)
print(decompose_ts_mul.seasonal)
print(decompose_ts_mul.resid)
print(decompose_ts_mul.observed)
decompose_ts_mul.plot()

#If you can observe the diffrence these plots
#Now lets us plot ACf plot to check the auto correlation
# ACF and PACF plot on Original data sets 
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(cocacola.Sales, lags = 4)
#we can observe the output in which r1,r2,r3 and r4 has higher correlation
#This is all about EDA
# ACF is an (complete) auto-correlation function gives values 
# of auto-correlation of any time series with its lagged values.
tsa_plots.plot_pacf(cocacola.Sales, lags=4)
# PACF is a partial auto-correlation function. 
# It finds correlations of present with lags of the residuals of the time series

#Lets us apply data to data driven model
# Simple Exponential Method
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
ses_model = SimpleExpSmoothing(Train["Sales"]).fit()
pred_ses = ses_model.predict(start = Test.index[0], end = Test.index[-1])
#Now calculating MAPE
MAPE(pred_ses,Test.Sales) 
#we are getting 8.30789774829734
# Holt winter exponential smoothing
hw_model = Holt(Train["Sales"]).fit()
pred_hw = hw_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hw, Test.Sales) 
#we are getting 9.80941173594165

# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Sales"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_add_add, Test.Sales) 
#1.5023826355347967

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"], seasonal = "mul", trend = "add", seasonal_periods = 4).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_mul_add, Test.Sales) 
#2.8863516520618875

#Let us apply to complete dataset of cocacola
#we have seen that hwe_model_add_add has got lowest MAPE,hence it is selected
hwe_model_add_add = ExponentialSmoothing(Train["Sales"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()
#import the new dataset for which prediction has to be done
new_data = pd.read_excel("CocaCola_Sales_New_Pred.xlsx")
newdata_pred=hwe_model_add_add.predict(start=new_data.index[0],end=new_data.index[-1])
MAPE(newdata_pred,Test.Sales)
newdata_pred




