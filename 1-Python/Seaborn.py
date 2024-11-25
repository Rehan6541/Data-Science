'''                29/04/24                          '''
import seaborn as sns
import pandas as pd
car=pd.read_csv('Cars.csv')
car.head()
car.columns
sns.relplot(x='HP',y='MPG',data=car)
sns.relplot(x='HP',y='MPG',data=car,kind='line')
sns.relplot(x='HP',y='MPG',data=car,kind='line')
#Boxplot
sns.catplot(x='HP',y='MPG',data=car,kind='box')
#Histogram
sns.distplot(car.HP)
#info about the data
car.describe()
#Graphical representation
import matplotlib.pyplot as plt
import numpy as np
plt.bar(height=car.HP,x=np.arange(1,82,1))
sns.distplot(car.HP)
#Boxplot
sns.catplot(data=car,kind='box')
sns.distplot(car.MPG)
sns.distplot(car.VOL)
sns.distplot(car.SP)
sns.boxplot(car.SP)
#joint plot,sns.jointplot
#histogram
sns.jointplot(x=car['HP'],y=car['MPG'])
#count plot
plt.figure(1,figsize=(16,10))
sns.countplot(car['HP'])










