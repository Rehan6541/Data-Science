#Created on Wed Jul 31 08:33:28 2024

#Data preprocessing
import pandas as pd
#let us import the dataset
df=pd.read_csv("ethnic diversity.csv")
#check the data types
df.dtypes
#salary data type is float change it to int
df.Salaries=df.Salaries.astype(int)
df.dtypes
#now the salary data is in int
#age data is in int it should be in float
df.age=df.age.astype(float)
df.dtypes

#Identify the duplicate
df_new=pd.read_csv("education.csv")
duplicate=df_new.duplicated()
#output of this function is single column
#if there is duplicate records output is true
#if there is no duplicate records output is false
#Series will be created
duplicate
sum(duplicate)
#output will be 0
#Now let us import another dataset
df_new1=pd.read_csv("mtcars_dup.csv")
duplicate1=df_new1.duplicated()
duplicate1
sum(duplicate1)
#There are 3 duplicates records
#row 17 is duplicate of row 2 like wise yoou can 3 duplicate records
#there is function called drop_duplicates()
#which will drop all duplicates
df_new2=df_new1.drop_duplicates()
duplicate2=df_new2.duplicated()
duplicate2
sum(duplicate2)


#outliers treatement
import pandas as pd
import seaborn as sns
df=pd.read_csv("ethnic diversity.csv")
#now let us find outliers in salaries
sns.boxplot(df.Salaries)
#There are outliers
#let us check outliers in age column
sns.boxplot(df.age)
#there are no outliers
#let us calculate outliers
IQR=df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)
#have observed IQR in variable explorer
#no,because IQR is in capital letters
#treated as constant
IQR

###########01/08/24
#but if we will try as I,Iqr or iqr then it is showing
I=df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)
Iqr=df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)
iqr=df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)

lower_limit=df.Salaries.quantile(0.25)-1.5*IQR
upper_limit=df.Salaries.quantile(0.75)+1.5*IQR
#now if you willl check lower limit of salarie it is -19446.9675
#upper limit is 93992.8125
#there is negative salary so   make it 0
#go to variable exploere and make it zero

#Trimming
import numpy as np
outliers_df=np.where(df.Salaries>upper_limit,True,
                     np.where(df.Salaries<lower_limit,True,False))
#you can check outliers of column in variable explorer
df_trimmed=df.loc[~outliers_df]
df.shape
#(310,13)
df_trimmed.shape
#(306,13)
sns.boxplot(df_trimmed.Salaries)

#Replacement technique
#Drawback of trimming technique is we are losing the data
df=pd.read_csv("ethnic diversity.csv")
df.describe()

#record no 23 has got outliers
#map all the outlier values to upper limit
df_replacement=pd.DataFrame(np.where(df.Salaries>upper_limit,upper_limit,np.where(df.Salaries<lower_limit,lower_limit,df.Salaries)))
#if the values are greater than the upper limit
#map it to upper limit and less the lower limit
#map it to lower limit,if it is within range
#then keep as it is
sns.boxplot(df_replacement[0])


##Winsorizer
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Salaries'])
#copy  winsorizer and paste in help tab of
#top right window study the method
df_t=winsor.fit_transform(df[['Salaries']])
sns.boxplot(df['Salaries'])
sns.boxplot(df_t['Salaries'])

###########02/08/24
#zero variance and near zero variance
#if there is no variance in the feature the ml 
#will not get any intelligence its better to ignore its feature
import pandas as pd
df=pd.read_csv("ethnic diversity.csv")
df.info()
df.Salaries.var()==0
'''
EMP ID      False
ZIP         False
Salaries    False
age         False
'''

#here EmpId and ZIP is nominal data
#salary has  444195346.7421577  which is not close to 0
#similarly age
df.var(axis=0)==0
#error, so we have to check column wise

import numpy as np
df=pd.read_csv("D:/data science/5-Data_prep/modified ethnic.csv")
#check for null values
df.isna().sum()

import pandas as pd
import numpy as np
df=pd.read_csv("modified ethnic.csv")
#check for null values
df.isna().sum()

#create an imputer that cretaes Nan values
#mean and median is used for numeric data
#mode is used for dicrete data(position,sex,maritalDes)
from sklearn.impute import SimpleImputer
mean_imputer=SimpleImputer(missing_values=np.nan,strategy="mean")
#check the dataframe
df['Salaries']=pd.DataFrame(mean_imputer.fit_transform(df[['Salaries']]))
#check the dataframe
df['Salaries'].isna().sum()

#Median imputer
median_imputer=SimpleImputer(missing_values=np.nan,strategy="median")
#check the dataframe
df['age']=pd.DataFrame(mean_imputer.fit_transform(df[['age']]))
#check the dataframe
df['age'].isna().sum()

#Mode imputer
mode_imputer=SimpleImputer(missing_values=np.nan,strategy="most_frequent")
#check the dataframe
df['Sex']=pd.DataFrame(mode_imputer.fit_transform(df[['Sex']]))
#check the dataframe
df['Sex'].isna().sum()

df['MaritalDesc']=pd.DataFrame(mode_imputer.fit_transform(df[['MaritalDesc']]))
#check the dataframe
df['MaritalDesc'].isna().sum()

import numpy as np
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

#step 1: generate an imbalanced dataset
X,y=make_classification(n_samples=1000, n_features=20,n_informative=2,n_redundant=10, n_clusters_per_class=1,weights=[0.99],flip_y=0,random_state=1)
'''
Parameters
n_sampels=1000:
        The total number of samples(datapoints) to generate.
        Here, 1000 sampples will be created
    
    n_features=20:
        The total number of features(columns) in the dataset
        Each sample will have 20 features
        
    n_informative=2:
        Number of informative features
        These features are udeful for prediictinf the target variables
        
    n_redundant=10:
        The number of redundant features
        These features are generated as random linear combinations
         
    n_clusters_per_class=1:
        The number of clusters per class
        Each class will have 1 cluster of points
        This parameter is useful for controlling the overlap between classes
    
    weights=[0.99]:
        the proportions of samples assigned to each class
        Here, 99% of samples will belong to one class
'''

#show the original class distribution
print("Original class distribution: ", np.bincount(y))

#step 2: Apply SMOTE to balance the dataset
smote=SMOTE(random_state=42)
X_res,y_res=smote.fit_resample(X,y)

#show the new class distribution after applying smote
print("Resampled class distribution: ", np.bincount(y_res))


----------------------------------------------------
###############05/08/24
#show the original class distribution
print("Original class distribution:{np.bincount(y)}")

from sklearn.model_selection import train_test_split
#step2:split the data into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

#step3:Apply SMOTE to balance the training dataset
smote=SMOTE(random_state=4)
X_train_res,y_train_res=smote.fit_resample(X_train,y_train)

#show the new class distribution after applying SMOTE
print(f"Resampled class distribution :{np.bincount(y_train_res)}")

#step4:Train a classifier on the balanced dataset
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(random_state=42)
clf.fit(X_train_res,y_train_res)

#step5:Evaluate the classifier on the test set
y_pred=clf.predict(X_test)
from sklearn.metrics import confusion_matrix

print("Confusion Matrix")
print(confusion_matrix(y_test,y_pred))

#####log transformation technique
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#generate sample dataset
np.random.seed(0)
data=np.random.exponential(scale=2.0,size=1000)
df=pd.DataFrame(data,columns=['Value'])

#perfrom log transformation
df['logvalue']=np.log(df['Value'])

#plot the original and log_transformed data
fig,axes=plt.subplots(1,2,figsize=(12,6))
                      
# Original data
axes[0].hist(df['Value'], bins=30, color='blue', alpha=0.7)
axes[0].set_title('Original Data')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

#log-Transformed data
axes[1].hist(df['logvalue'], bins=30, color='green', alpha=0.7)
axes[1].set_title('Log Tranformed Data')
axes[1].set_xlabel('Log(Value)')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
                      
                      




















