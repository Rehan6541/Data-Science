# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 09:19:53 2024

@author: Hp
"""

5. Given a data of house prices [200000, 250000, 150000, 350000, 300000, 
400000, 450000, 600000, 650000, 500000, 550000]. Calculate the following:
The median of the dataset.
The 25th percentile (1st quantile), 50th percentile (2nd quantile, also the 
median), and 75th percentile (3rd quantile).
Visualize the data using a box plot.
--->>>
import pandas as pd
import matplotlib.pyplot as plt
prices=[200000, 250000, 150000, 350000, 300000, 400000, 450000, 600000, 650000, 500000, 550000]
df=pd.DataFrame(prices)
df.quantile(0.25)
df.quantile(0.50)
df.quantile(0.75)
plt.boxplot(df)


4. Given a dataset containing various types of data, categorize each 
variable into the appropriate statistical data type: Nominal, Ordinal, 
Interval, or Ratio. Then, write code to demonstrate how you would work 
with each type of data.
Example Dataset:
ID Name Age Education 
Level
Salary Joining 
Year
1 Sophie 22 Bachelor's 60000 2022
2 Aryan 25 Master's 75000 2020
3 Amit 28 PhD 78000 2018
4 Charu 26 Bachelor's 45000 2015
5 Piyush 37 Master's 92000 2010
--->>>
ID-
Name- Nominal
Age-
Education Level- Categorical
Salary-
Joining Year-Interval
import pandas as pd
df=pd.DataFrame()
id=[1,2,3,4,5]
name=['Sophie','Aryan','Amit','Charu','Piyush']
age=[22,25,28,26,37]
edu=["Bachelor's","Master's","PHD","Bachelor's","Master's"]
salary=[60000,75000,78000,45000,92000]
year=[2022,2020,2018,2015,2010]
df['ID']=id
df['Name']=name
df['Age']=age
df['Education']=edu
df['Salary']=salary
df['Year']=year

set[df['ID']]















3. Generate 1,000 random values following a logarithmic distribution with 
a probability parameter p = 0.3. Perform the following tasks:
Plot the histogram of the dataset.
Calculate the mean of the dataset.
Overlay the probability mass function (PMF) of the logarithmic 
distribution on the histogram.
--->>>
import numpy as np
mean=0
std=0.3
values=np.random.lognormal(mean,std,1000)
df=pd.DataFrame(values)
plt.hist(values)
df.mean()













2. Generate a dataset of 1,000 random values generated from a lognormal 
distribution with a mean of 0 and a standard deviation of 1 in the log-space, 
perform the following tasks:
Plot the histogram of the dataset.
Calculate the mean and median of the dataset.
Fit a lognormal distribution to the data and overlay the probability density 
function (PDF) on the histogram.
--->>>
import numpy as np
std=1
mean=0
values=np.random.lognormal(mean,std,1000)
df=pd.DataFrame(values)
plt.hist(values)
df.mean()
df.median()













1. Given a dataset of integers or floating-point numbers, calculate the 
following descriptive statistics:
Mean
Median
Mode
Variance
Standard Deviation
Sample Dataset: [20, 40, 40, 40, 30, 50, 60]
--->>>
Sample_Dataset=[20, 40, 40, 40, 30, 50, 60]
df=pd.DataFrame(Sample_Dataset)
df.mean()
df.median()
df.mode()
df.var()
np.std(df)

