# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 09:17:26 2024

@author: Hp
"""
1.You are given a dataset with two numerical features Height and Weight. 
Your goal is to cluster these people into 3 groups using K-Means clustering. 
After clustering, you will visualize the clusters and their centroids.
1.Load the dataset (or generate random data for practice).
2.Apply K-Means clustering with k = 3.
3.Visualize the clusters and centroids.
4.Experiment with different values of k and see how the clustering changes.

import pandas as pd
df=pd.read_csv("C:\Data science\clustering test_13_09\HeightWeight.csv")
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
df.head()
df.columns
df.shape
df.describe()


scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

df_scaled = pd.DataFrame(df_scaled, columns=df.columns)


km = KMeans(n_clusters=3, random_state=42)
y_predicted = km.fit_predict(df_scaled)

df['cluster'] = y_predicted

df.head()

centroids_original = scaler.inverse_transform(km.cluster_centers_)

km.cluster_centers_

df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

# Plotting each cluster with different colors
plt.scatter(df1['Height(Inches)'], df1['Weight(Pounds)'], color='green', label='Cluster 1')
plt.scatter(df2['Height(Inches)'], df2['Weight(Pounds)'], color='red', label='Cluster 2')
plt.scatter(df3['Height(Inches)'], df3['Weight(Pounds)'], color='black', label='Cluster 3')

# Plotting the cluster centers
#plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='purple', marker='*', label='Centroid')
plt.scatter(centroids_original[:, 0], centroids_original[:, 1], color='purple', marker='*', s=200, label='Centroid')


# Labeling the axes and showing the legend
plt.xlabel('Height(Inches)')
plt.ylabel('Weight(Pounds)')
plt.legend()
plt.show()

2. You have a dataset of customers with features Age, Annual Income, and 
Spending Score. You need to apply hierarchical clustering to segment these 
customers. Plot a dendrogram to decide the optimal number of clusters and 
compare it with K-Means clustering results.
Steps:
 Load the dataset.
 Apply hierarchical clustering.
 Plot a dendrogram and choose the number of clusters.
 Apply K-Means clustering with the same number of clusters.
 Compare the results.


import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("C:\Data science\clustering test_13_09\Mall_Customers.csv")
a=df.describe()
df1=df.drop(["CustomerID"],axis=1)

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm=norm_func(df1.iloc[:,1:])

b=df_norm.describe()

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(df_norm,method="complete",metric="euclidean")
plt.figure(figsize=(15,8))
plt.title("Hireraarchical clustering dendogram")
plt.xlabel("Index")
plt.ylabel("Distance")

sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()

from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=3,linkage='complete',metric="euclidean").fit(df_norm)
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)
df1['clust']=cluster_labels
df=df1.iloc[:,[3,0,1,2]]
df.iloc[:,2:].groupby(df1.clust).mean()

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

z=linkage(df_scaled2,method="complete",metric="euclidean")
plt.figure(figsize=(15,8))
plt.title("Hirachical Clustering")
plt.xlabel("Index")
plt.ylabel("Distance")

sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()










































































 




plt.scatter(df['Age'], df['Income'])
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()

# Preprocessing using Min-Max Scaler
scaler = MinMaxScaler()

# Fit and transform the selected columns
df_scaled = scaler.fit_transform(df[columns_to_cluster])

# Convert the scaled data back to a DataFrame for easier handling
df_scaled = pd.DataFrame(df_scaled, columns=columns_to_cluster)

# Initialize KMeans
km = KMeans(n_clusters=3, random_state=42)
y_predicted = km.fit_predict(df_scaled)

# Add the cluster labels to the original dataframe
df['cluster'] = y_predicted

# Display the first few rows of the updated dataframe
print(df.head())

# Display cluster centers in the scaled space
print("Cluster Centers (scaled):")
print(km.cluster_centers_)

# Inverse transform the cluster centers back to the original scale
centroids_original = scaler.inverse_transform(km.cluster_centers_)
print("Cluster Centers (original scale):")
print(centroids_original)

# Creating dataframes for each cluster
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

# Plotting each cluster with different colors (Age vs Income as an example)
plt.scatter(df1['Age'], df1['Income'], color='green', label='Cluster 1')
plt.scatter(df2['Age'], df2['Income'], color='red', label='Cluster 2')
plt.scatter(df3['Age'], df3['Income'], color='black', label='Cluster 3')

# Plotting the cluster centers
plt.scatter(centroids_original[:, 1], centroids_original[:, 4], color='purple', marker='*', s=200, label='Centroid')
plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()
plt.show()






