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

#Objective: To determine if there is a significant difference between the two promotional offers
offers=pd.read_excel('Promotion.xlsx')
offers.columns=['InterestRateWaiver','StandardPromotion']
offers.head()

#Normality Tests
print("InterestRateWaiver normality: ", stats.shapiro(offers.InterestRateWaiver))
print("StandardPromotion normality: ", stats.shapiro(offers.StandardPromotion))

#Variance test
levene_test=scipy.stats.levene(offers.InterestRateWaiver,offers.StandardPromotion)
print("Levene's test results: ",levene_test)

#p value=0.2875
#H0: Variance equal
#H1: Variance not equal
#p value>0.05 fail to reject null hypothesis

#Two sample t test
ttest_result=stats.ttest_ind(offers.InterestRateWaiver,offers.StandardPromotion)
print("Two-Sample T-test :",ttest_result)
#Result:pvalue=>0.02422584468584315
#Interpretation
#HO:the offer has same mean impact
#H1:the mean impacts of the two offers are different
#Since the pvalue (0.02422584468584315) is less than 0.05,we reject the null hypothesis
#Conclusion: there is a signifcant difference between the two promotional offers

#Mood's Median Test
#Objective Is the median of pooh.piglet and tigger are statistically equal
#It has equal median or not
animals=pd.read_csv("animals.csv")
animals.describe()

#normality test
print("Pooh normality test",stats.shapiro(animals.Pooh))
#pvalue=0.012278728187084198 reject the H0
print("Piglet normality test",stats.shapiro(animals.Piglet))
#pvalue=0.04488762468099594 reject the H0
print("Tigger normality test",stats.shapiro(animals.Tigger))
#pvalue=0.021947985514998436 reject the H0
#H0:data is normal
#H1:data is not naormal
#Since the pvalue  is less than 0.05,we reject the null hypothesis
#Data is not normal hence mood's test

#Median Test
median_test_result=stats.median_test(animals.Pooh,animals.Piglet,animals.Tigger)
print("Mood's Median Test:",median_test_result)
#Result:pvalue=>0.186
#Interpretation
#H0:All groups have equal medians
#H1:All groups have significant diffrence between median
#Since the pvalue  is grater than 0.05,we fail to reject the null hypothesis
#Conclusion data is normal

#One way ANOVA
#Objective: is the tranctaion time same for the three suppliers are not
#Siggnificant difference
contract=pd.read_excel("ContractRenewal_Data(unstacked).xlsx")
contract.columns=["Supp_A","Supp_B","Supp_C"]

#normality test
print("Supp_A normality test",stats.shapiro(contract.Supp_A))
#pvalue=0.8961992859840393 fail to reject the H0
print("Supp_B normality test",stats.shapiro(contract.Supp_B))
#pvalue=0.6483432650566101 fail to reject the H0
print("Supp_C normality test",stats.shapiro(contract.Supp_C))
#pvalue=0.5719417929649353 fail to reject the H0
#All value are grater than 0.05 we fail to reject the null hypothesis
#H0 is accepted mean data is normal
#H0:data is normal
#H1:data is not naormal

#Variance TEST
levene_test=scipy.stats.levene(contract.Supp_A,contract.Supp_B,contract.Supp_C)
print("Levenue test (variance):",levene_test)
#H0 : data is having equal variance
#H1 : data is having significant diggrence variance
#p value=0.7775071819400866,H0 is accepted

#Anova Test
anova_result=stats.f_oneway(contract.Supp_A,contract.Supp_B,contract.Supp_C)
print("One way anova test:",anova_result)
#P value 0.10373295731933224
#Interpretation
#H0:all suppliers have the same mean tranction time
#H1:at least one supplier has a diffrent mean
#Since the pvalue  is greater than 0.05,we fail to reject the null hypothesis
#H0 is accepted
#Conclusion the tranction time for the three suppliers are not significant diffrent

##Two Proportion Z-Test

#use a two sided test when you wnat to detect the diffrence without 
#assuming beforehand which gruop will ahe haigher or lower proportion
#Example-Testing if there is signifivant diffrence in soft drink consumption
#between adults and children.
#Objective-There is significance diffrent consumption of soft drink between adults and children

soft_drink=pd.read_excel("JohnyTalkers.xlsx")
from statsmodels.stats.proportion import proportions_ztest

#Data preparation
count=np.array([58,152])
nobs=np.array([450,740])
#The two proportion ztest compares the proportions of the two groups

#count=np.array([58,152])-The number of success
#people consuming soft drinks in each group adult or children

#nobs=np.array([450,740])-the total number of observation
#in each group (total adults and children survyed)
#The count and nobs value

#Smilarly if 740 children were suryed and 152 of them reported consuming 
#soft drinks the second count is 152
#thus count=np.array([58,152])

#Total number of adults suryed is 480 
#the total number of children suryed is 740
#hence nobs=np.array([450,740])

#these values are often extracted from a dataset
#if your data is in a file(like jhonytalker)
#you can calculate these values as follows

import pandas as pd

#Load the dataset
soft_drink_data=pd.read_excel("JohnyTalkers.xlsx")

#filter the data into adults abd children categories
adults=soft_drink_data[soft_drink_data['Person']=='Adults']
children=soft_drink_data[soft_drink_data['Person']=='Children']

#Count os uccess (soft drink consumer) for each group
count_adults=adults[adults['Drinks']=='Purchased'].shape[0]
count_children=children[children['Drinks']=='Purchased'].shape[0]

#Total observataion for each group
nobs_adults=adults.shape[0]
nobs_children=children.shape[0]

#Final arrays for Ztest
count=[count_adults,count_children]
nobs=[nobs_adults,nobs_children]

print("Counts(soft drink consumer):",count)
print("Total Observation:",nobs)

#Two sided test
z_stat,p_val=proportions_ztest(count, nobs,alternative='two-sided')
print("Two-sided proportion Test:",z_stat,"p-value:",p_val)
#result p_value-0.000
#interpretation
#H0:Proportion of adults and children consuming the soft drink are
#H1: propoertion are diffrent
#Since the pvalue  is less than 0.05,we reject the null hypothesis
#H1 is accepted
#Conclusion= there is a significant diffrent in soft drink consumption


##Chi-Square Test
#Objective-is defective proportion are independent of the country?
#The dataset contains two columns:
    
#Defective:indicates whether an item is defective(likely binary,
#with 1 for defective and 0 for not dfective)
#country:specifies the country associated wih the item(eg.:Indai)
#the dataset has 800 entries and there are
#no missing values in either column it appears to be desighned
#to analysis defect rate across diffrent countries
#which alignhs to the chi square test which you didi earlier
#to determine if defetiveness is independebnt of the country

Bahaman=pd.read_excel("Bahaman.xlsx")

#Crosstabulation
count=pd.crosstab(Bahaman["Defective"],Bahaman["Country"])
count

#Chi-Square Test
chi2_result=scipy.stats.chi2_contingency(count)
print("Chi-Square test:",chi2_result)
#Result pvalue is 0.6315243037546223
#Interpretation
#H0:defective proportion is independent of the county
#H1:defective proportin is dependent of the country
#Since the pvalue  is greater than 0.05,we fail to reject the null hypothesis
#H0 is accepted
#Conclusion: The defective items are independent of the countries





