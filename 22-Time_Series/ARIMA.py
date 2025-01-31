# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:42:17 2025

@author: Hp
"""

import pandas as pd
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot

'''
pandas as pd: import the pandas library for handling and anallyzing struct
tsa_plots: imports the time-series plotiing funnctions as ACF(Autocorrelation Function)
and PACF (Partial Autocorrelation Function ).
ARIMA: imports the ARIMA model for time-series forecasting.
mean_squared_error: used for evaluating forecast
'''

walmart=pd.read_csv("Walmart_Footfalls_Raw.csv")
#Data Partition
Train=walmart.head(147)
Test=walmart.tail(12)

tsa_plots.plot_acf(walmart.Footfalls, lags=12)
tsa_plots.plot_pacf(walmart.Footfalls, lags=12)

'''
When analyzinng ACF and PACF plots, we follow these rules:
    
    AR order (p) from PACF:
        look at the partial Autocorrelation Funnction (PACF) plot.
        The number of significant lags before the PACF drops near to zero
        suggest the AR order.
        If PACF show a sharp cutoff after lag 4, we take AR(4).
        
        
    MA order (q) from ACF:
        look at the autocorrelation Function (ACF) plot.
        The number signic=ficant lags before the ACF drops to near zero
        suggests.
        If ACF shows a sharp cutoff after lag 6 , we take MA(6).
        
        
First you try p=4 and q=4
Then you try p=4 and q=6
'''

#ARIMA withn AR=4,MA=6
model1=ARIMA(Train.Footfalls,order=(4,1,6))
res1=model1.fit()
print(res1.summary())
'''
Creates an ARIMA model with :
AR(Auto-Regression) term =4
I(integrated) term =1 (indicates firdt-order diffrencing to make  the dataset)
MA(MOving Average) term=6

When to use Differncing(d in ARIMA)
Before selecting p and q,ensure the time series is stationary:
    
If the mean and variance change over time apply diffrencing
Use the augemnted Dickey Filler (ADF) test to chech  for the stationary
If the series ins not  stati0nary apply first order diffrencing(d=1)

I p-value>0.05,the series is not statinary so apply diffrencng(d=1)
'''
from statsmodels.tsa.stattools import adfuller
result=adfuller(walmart.Footfalls)
print('ADF Statistic:',result[0])
print("p value:",result[1])
#p value: 0.9342202042969283
#data is not stationary
#take d=1

#forecast for nect 12 months
start_index=len(Train)#begins prediction after the training dtatset
end_index=start_index+11#predicts the next 12 periods
forecast_test=res1.predict(start=start_index,end=end_index)#generates forecats
print(forecast_test)

#evaluate forecasts
rmse_test=sqrt(mean_squared_error(Test.Footfalls,forecast_test))
print('Test RMSE:%.3f'%rmse_test)

#plot forecasts against actual outcomes
pyplot.plot(Test.Footfalls)
pyplot.plot(forecast_test,color='red')
pyplot.show()

#Auto-ARIMA=Automatically discover the optimal order for an ARIMA model
#pip install pmdarima --user
'''
pmdarima is an auto ARIMA pacakage that automatically selects the best(p,d)
start_p=0,start_q=0:Initial values for AR and MA terms
max_p=12,max_q=12:Maximum values for AR and MA
m=1:indicates a non seasonal model
d=None:Automatically determines the diffrencing order
seasonal=False:Disables seasonal components
trace=True:Displays the selection process
stepwise=True:Uses a stepwise appraoch for efficiency
'''

import pmdarima as pm
ar_model=pm.auto_arima(Train.Footfalls,start_p=0,start_q=0,
                       max_p=12,max_q=12,#maximum p and q
                       m=1,#frequency of series
                       d=None, #let determine 'd'
                       seasonal=False, #NOseasonability
                       start_P=0,trace=True,
                       error_action='warn',
                       stepwise=True)

#Best parameters ARMIA
#ARIMA with AR=3 ,T=1,MA=5
model=ARIMA(Train.Footfalls,order=(3,1,5))
res=model.fit()
print(res.summary())

#forecast for next 12 monthns
start_index=len(Train)
end_index=start_index+11
forecast_best=res.predict(start=start_index,end=end_index)
print(forecast_best)

#evaluate forecasts
rmse_best=sqrt(mean_squared_error(Test.Footfalls,forecast_best))
print('Test RMSE:%.3f'%rmse_best)

#plot forecasts against actual outcomes
pyplot.plot(Test.Footfalls)
pyplot.plot(forecast_best,color='red')
pyplot.show()











        