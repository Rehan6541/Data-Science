# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 09:17:37 2024

@author: Hp
"""
import numpy as np
from numpy import array
from scipy.linalg import svd
A=array([[1,0,0,0,2],[0,0,3,0,0],[0,0,0,0,0],[0,4,0,0,0]])
#SVD
U,d,Vt=svd(A)
print(U)
print(d)
print(Vt)
print(np.diag(d))
#SVD Applying to a dataset
import pandas as pd
data=pd.read_excel("C:/9-PCA_SVD/University_Clustering.xlsx")
data.head()
data=data.iloc[:,2:]#for removing non numeric data
data.head()    
from sklearn.decomposition import TruncatedSVD
svd=TruncatedSVD(n_components=3)
svd.fit(data)
result=pd.DataFrame(svd.transform(data))
result.head()
result.columns="pc0","pc1","pc2"
result.head()
##Scatter plot
import matplotlib.pylab as plt
plt.scatter(x=result.pc0,y=result.pc1)
