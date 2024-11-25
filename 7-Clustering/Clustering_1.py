###Created on Thu Aug 8 2024
###CLUSTERING
import pandas as pd
import matplotlib.pyplot as plt
Univ1=pd.read_excel("University_Clustering.xlsx")
a=Univ1.describe()
#We have one column 'State' which really not useful we will drop it
Univ=Univ1.drop(["State"],axis=1)
'''We know that there is scale difference among the columns,
which we have to remove
whenever there is mixed data apply normalization'''
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
'''now apply this normalization functionto univ datframe 
for all the rows and columns from i until end
since 0 th column has University name hence skipped'''
df_norm=norm_func(Univ.iloc[:,1:])
'''You can check the df_norm dataframe which is scaled
between values of 0 to 1
you can apply describe function to new dataframe'''
b=df_norm.describe()
'''Before you apply clustering you need to plot dendrogram first
Now to create dendrogram we need to measure distance
we have to import linkage'''
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
#linkage function gives us hierarchical or aglomerative clustering
#ref the help for linkage
z=linkage(df_norm,method="complete",metric="euclidean")
plt.figure(figsize=(15,8))
plt.title("Hireraarchical clustering dendogram")
plt.xlabel("Index")
plt.ylabel("Distance")
#ref help of dendrogram
#sch.dendrogram(z)
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()
#dendrogram()
#Applying aglomerative clustering choosing 5 as clusters from dendrogram
#whatever has been displayed is dendrogram is not clustering
#it is just showing number of possible clusters
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=3,linkage='complete',metric="euclidean").fit(df_norm)
#apply labels to the clusters
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)
#Assign this series to the Univ dataframe as column and name the column
Univ['clust']=cluster_labels
#we want to relocate the column 7 to 0th position
Univ1=Univ.iloc[:,[7,1,2,3,4,5,6]]
#now check the Univ dataframe
Univ1.iloc[:,2:].groupby(Univ.clust).mean()
#from the output cluster 2 has got higest top10
#lowest accept  ratio,best faculty ratio and highest expenses
#highest graduate ratio
Univ.to_csv("Univ_clustering.csv",encoding="utf-8")
import os
os.getcwd




























 






