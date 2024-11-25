######Tue Aug 13 08:58:57 2024

#importing required lib
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
df=pd.read_csv("income.csv")
df.head()
plt.scatter(df.Age,df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')
km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(df[['Age','Income($)']])
y_predicted
df['cluster']=y_predicted
df.head()
km.cluster_centers_

#Creating dataframes for each clusters
df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]

#ploting each cluster with different colors
plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend()
#this do not show proper cluster because we have not scaled the data
#for proper cluster we have to scale the data

#Preprocessing using min max scaler
scaler=MinMaxScaler()

scaler.fit(df[['Income($)']])
df['Income($)']=scaler.transform(df[['Income($)']])

scaler.fit(df[['Age']])
df['Age']=scaler.transform(df[['Age']])
df.head()

plt.scatter(df.Age,df['Income($)'])

km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(df[['Age','Income($)']])
y_predicted

df['cluster']=y_predicted
df.head()
km.cluster_centers_

#Creating dataframes for each clusters
df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]

#ploting each cluster with different colors
plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend()

#######14/08/24
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
#lets try to undestand first how k means works for 2D data
#for that,generate random numbers in the range 0 to 1
#and with uniform probablity of 1/50
X=np.random.uniform(0,1,50)
Y=np.random.uniform(0,1,50)
#create a empty dataframe with 0 rows and 2 colmns
df_xy=pd.DataFrame(columns=["X","Y"])
#assign the values of X and Y to these columns
df_xy.X=X
df_xy.Y=Y
df_xy.plot(x="X",y="Y",kind="scatter")
model1=KMeans(n_clusters=3).fit(df_xy)

'''
with the data X and Y ,apply kmeans model,
generate scatter plot
with scale/font-10
cmap=plt.cm.coolwarm:cool combination
'''
model1.labels_
df_xy.plot(x="X",y="Y",c=model1.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm)

####################################################
###ELBOW CURVE
Univ1=pd.read_excel("c:/7-Clustering/University_Clustering.xlsx")
Univ1.describe()
Univ=Univ1.drop(["State"],axis=1)

#yoyu know tthat there is scale diffrence among the columns ,which we 
#either by  using mormalization or standardizartion

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

#now apply this normalization  function to univ datafram,e for all thye 

df_norm=norm_func(Univ.iloc[:,1:])
'''
what will be ideal cluster number, will it be 1,2 or 3
'''
TWSS=[]
k=list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    
    TWSS.append(kmeans.inertia_)#total sum of square
'''
Kmeans inertia also kowns as sum of square Errors  
(or SSE). calculates the sum of distances of all points within a cluster  from centroid of 
of the point . It is the difference betweeen the obserrved value and predicted value


'''
TWSS
#
plt.plot(k,TWSS,'ro-');
plt.xlabel("No_of_clusters");
plt.xlabel("Total_within_SS")
'''
How to select value of the elbow curve 
when k changes from  2 to 3, then decrease
in twss is higher than 
when k changes from 3 to 4 
when k values changes from 5 to 6 decrease 
'''

model=KMeans(n_clusters=3)
model.fit(df_norm)
model.labels_
mb=pd.Series(model.labels_)
Univ['clust']=mb
Univ.head()
Univ=Univ.iloc[:,[7,0,1,2,3,4,5,6]]
Univ
Univ.iloc[:,2:8].groupby(Univ.clust).mean()

Univ.to_csv("kmeans_university.csv",encoding="utf-8")
import os
os.getcwd














