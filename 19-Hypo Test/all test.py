# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 08:20:31 2025

@author: Hp
"""

import pandas as pd
import numpy as np
import scipy
from scipy import stats
#provides statistical functions
#stats contains a variety of statistical tests
from statsmodels.stats import descriptivestats as sd
#provider description statistics tools,including the sign_test
from statsmodels.stats.weightstats import ztest
#used for conducting z test on datasets

#1 Sample sign test
#whenever thereis a single sample and data is not normal
marks=pd.read_csv("Signtest.csv")

#Normal QQ plot
import pylab
stats.probplot(marks.Scores,dist='norm',plot=pylab)
#Creates a QQ plot for visual check if the data follows a normal distribution

#Test for normality
shapiro_test=stats.shapiro(marks.Scores)
#perform the shapiro-Wilk test for normality
#H0=the data is normally distributed
#H1=the data is not normally distributed

#outputs a test statistics and p=value
print(marks.Scores.describe())
#Mean=84.20000 and median=89.00000

#1 sample sign test
sign_test_result=sd.sign_test(marks.Scores,mu0=marks.Scores.mean())
print("Sign Test Result:",sign_test_result)
#Result p-value =0.8238029479980469
#Interpretation
#H0=the median of scores is equal to mean of scores
#H1=the median of scores is not equal to mean of scores
#Since the p value is greater than 0.5,we fail to reject the null hypothesis
#Conclusion=The mean and median of the scores is statistically similar but they are not similar as seen by the scores.describe()hence the test fails

##1 sample z test
fabric=pd.read_csv("fabric_data.csv")

#Normality tes
fabric_normality=stats.shapiro(fabric)
print("fabric normality is",fabric_normality)
#p value is 0.1460934281349182

fabric_mean=np.mean(fabric)
print("mean fabric length",fabric_mean)

#Z test
z_test_result,p_val=ztest(fabric["Fabric_length"],value=150)
print("Z Test Result:",z_test_result,"P value:",p_val)
#Result p-value =7.156241255356764e-06<0.05
#rejects the null hypothesis
#Interpretation
#H0=the mean of fabric length is exactly 150
#H1=fabric length is not exactly 150
#Since the p value is extremly small than 0.5,we reject the null hypothesis
#Conclusion=The mean fabric length significantly  differ from 150

#Mean whitney Test
fuel=pd.read_csv("mann_whitney_additive.csv")
fuel.columns=['Without_additive',"With_additive"]

#Normality test
print("Without additive normality:",stats.shapiro(fuel.Without_additive))
#p value=0.501198410987854>0.05 accepts the data is normal
print("With additive normality:",stats.shapiro(fuel.With_additive))
##p value=0.041048310697078705>0.05 rejects the data is normal

#Mann whitney test
mannwhitney_result=stats.mannwhitneyu(fuel.Without_additive, fuel.With_additive)
print("Mann whitney test result:",mannwhitney_result)
#Result:p value is 0.4457311042015709
#Interpretation
#H0=no diff in performance between without additive and with additive
#H1=significant diff exists
#Since the p value is  0.4457311042015709 is grater than 0.05 we fail to reject the null hypothesis
#Conclusion=adding fuel additive does not significantly impact performance
#Applies the mann whitney test check if there is significnt diff
#H0=the diff in performance of two groups
#H1=significant diff in performance

#Paired t test
#objective=to check there is diff between transaction time of supp A and Supp b
sup=pd.read_csv("paired2.csv")

#normality test
print("supplier A normality test",stats.shapiro(sup.SupplierA))
#pvalue=0.8961992859840393 fails to reject the H0,dtat is normal
print("supplier A normality test",stats.shapiro(sup.SupplierB))
#pvalue=0.8961992859840393 fails to reject the H0,dtat is normal

#Paired t test
t_test_result, pvalue=stats.ttest_rel(sup['SupplierA'],sup['SupplierB'])
print("Paired T-test results: ",t_test_result,"\np-value: ",pvalue)
#Result
#p-value=0.0
#H0: there is no significant differnce between transaction times of supplier A and supplier B
#H1: There is no difference




