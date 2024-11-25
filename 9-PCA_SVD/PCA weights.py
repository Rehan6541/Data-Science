# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 08:34:15 2024

@author: Hp
"""

import pandas as pd
import numpy as np
Univ1=pd.read_excel("C:/9-PCA_SVD/University_Clustering.xlsx")
Univ1.describe()
Univ1.info()
Univ=Univ1.drop(["State"],axis=1)
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
from sklearn.preprocessing import scale
Univ.data=Univ.iloc[:,1:]
#normalizing numerical data
uni_normal=scale(Univ.data)
uni_normal
pca=PCA(n_components=6)
pca_values=pca.fit_transform(uni_normal)
var=pca.explained_variance_ratio_
var
#PCA Weights
pca.components_
pca.components_[0]
#to check the cumulative variance
var1=np.cumsum(np.round(var,decimals=4)*100)
var1




















