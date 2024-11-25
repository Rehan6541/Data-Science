###########02/04/24
####PYTHON_FOR_DATA_SCIENCE
#pandas,numpy,seaborn,matplotlib

##SERIES->
#It is used to model one dimensional data,similar to a list in python
#The serirs obj also has a few more bit of data including index & name


import pandas as pd
songs2=pd.Series([145,142,5,20],name='counts')
#it is easy to inspect the index of a series (or data field)
songs2.index
#the index can be in a string form also.
#it may be in any form numbers,dates,strings and many more

songs3=pd.Series([145,142,5,20],name='counts',index=['paul','john','george','ringo'])
songs3.index
songs3

#the NaN Value
'''index     x1
1          a
2          b
3              ->NaN
4          d
this value stands for not a number and is usually ignonored 
in artithmatic operations.similar tu null in sql
if you load data from csv file,an empty value for an otherwise'''

#numeric column will become null
import pandas as pd
f1=pd.read_csv('age.csv')
f1

df=pd.read_excel("Bahaman.xlsx")
#None,NaN,nan and null are synonyms
#the series abj behaves similarly to a NumPy array

import numpy as np
numpy_ser=np.array([145,142,5,20])
songs3[1]
#142
numpy_ser[1]
#they both have same methods in common
songs3.mean()
numpy_ser.mean()


#the pandas series data structure provides support for the basic crud
#operation ->create,read,update and delete.

#Creation
george=pd.Series([10,7,1,22],
index=['1968','1969','1970','1970'],
name='George_songs')
george

'''the previous eg illustrates an intresting feature of pandas->
the values are strings and they are not unique .this can cause 
some confusion,but can also be useful when duplicate index are needed''' 
##################################################################

#Reading
#to read or select the data from a series
george['1968']
#we can iterate over data in a series as well.when iterating over a series
for item in george:
    print(item)
##################################################################    

#Updating
'''updating values in a series is can be a little tricky as well
for a given index label,the standard index assignment operation works'''
george['1970']=68
george['1969']=19
george


#deletion
#the del statement appears to have problem with duplicate index
s=pd.Series([2,3,4],index=[1,2,3])
del s[1]
s
#################################################################

##Convert types
import pandas as pd
song_66=pd.Series([3,6.0,11,9],
                  index=['George','Ringo','John','Paul'],
                  name="Counts")
song_66.dtypes
#dtypes("floated")
pd.to_numeric(song_66.apply(str))
#there will be error
pd.to_numeric(song_66.astype(str),errors='coerce')
#if we pass error='coerce' we can see thet it supports many formats
song_66.dtypes

##Dealing with None
#the filling method will replace them with the given value
song_66=song_66.fillna(-1)
song_66=song_66.astype(int)
#song_66=pd.to_numeric(song_66.apply(str))
song_66.dtypes

#NaN values can be dropped from the series using .dropna()
song_66=pd.Series([3.0,None,11.0,9.0],
                  index=['George','ringo','john','paul'],
                  name="counts")
song_66=song_66.dropna()
song_66

#Append,combine and joining two series
song_66=pd.Series([3.0,None,11.0,9.0],
                  index=['George','ringo','john','paul'],
                  name="counts")
song_69=pd.Series([5,20,50,200],
                  index=['ram','sham','jay','hari'],
                  name="Counts")
#to concate two series together simply we use append
song=pd.concat([song_66,song_69])
song
###############################################################

#Ploting Series
#Line graph
import matplotlib.pyplot as plt
fig=plt.figure()
song_69.plot()
plt.legend()

#Bar graph
fig=plt.figure()
song_69.plot(kind='bar')
song_66.plot(kind='bar',color='r')
plt.legend()

#Histogram
import numpy as np
data=pd.Series(np.random.randn(500),name='500_random')
fig=plt.figure()
ax=fig.add_subplot(111)
data.hist()


################################################################
#04/04/24
import pandas as pd
pd.__version__

#create using constructor
#create pandas dataframe from list
import pandas as pd
tech=[['spark',2000,'30days'],
     ['pandas',2000,'40days']]
df=pd.DataFrame(tech)
print(df)
col_name=['courses','fees','duration']
row_nam=['a','b']
df=pd.DataFrame(tech,columns=col_name,index=row_nam)
print(df)
df.dtypes

#you can also assign custom data types to columns.
#set custom types to dataframe
import pandas as pd
tech={
      'courses':['spark','pyspark','hadoop','python','pandas'],
      'fees':[2000,3000,4000,5000,6000],
      'duration':['30days','40days','50days','60days','70days'],
      'discount':[1.2,2.3,3.4,4.5,5.6]
      }
row_nam=['a','b','c','d','e']
df=pd.DataFrame(tech,index=row_nam)
print(df)
df.dtypes

#convert all types to best possible types
df2=df.convert_dtypes()
print(df2.dtypes)

#change all colums to same type
df=df.astype(str)
print(df.dtypes)

#change type for one or multiple columns
df=df.astype({'fees':int,'discount':float})
print(df.dtypes)

#convert data types for all columns in a list
df=pd.DataFrame(tech)
df.dtypes
cols=['fees','discount']
df[cols]=df[cols].astype('float')
df.dtypes

#Ignore errors
df=df.astype({'courses':int},errors='ignore')
df.dtypes

#generate error
df=df.astype({'courses':int},errors='raise')
df.dtypes

#convert fees column to numeric type
df=df.astype(str)
print(df.dtypes)
df['discount']=pd.to_numeric(df['discount'])
df.dtypes

#convert dataframe to csv
df.to_csv('data_file.csv')

#import dataframe csv
df=pd.read_csv('data_file.csv')
df

#pandas dataframe basic operation
#create dataframe with None/null to work with examples
import numpy as np
import pandas as pd
tech={
      'courses':['spark','pyspark','hadoop',None,'pandas'],
      'fees':[20000,3000,np.nan,5000,6000],
      'duration':['30days','','50days','60days','70days'],
      'discount':[1.2,2.3,3.4,4.5,5.6]
      }
row_nam=['a','b','c','d','e']
df=pd.DataFrame(tech,index=row_nam)
print(df)
df.dtypes

#################################################################
####05/04/24
#Dataframe properties
df.shape
#(5,4)
df.size
#20
df.columns
df.columns.values
df.index
df.dtypes
df.info

#Accessing one column content
df['fees']
#Accessing two column content
df[['fees','duration']]
#another method 
cols=['fees','duration']
df[cols]

#select certain rows and assign it to another dataframe
df2=df[3:]
df2=df[:3]

#accessing certain cell from columns
df['duration'][3]

#substracting specific value from column
df['fees']=df['fees']-1
df['fees']

#Pandas to manipulate Dataframe
#describe Dataframe
#describe Dataframe for all numeric columns
df.describe()
#it will show 5 number summary

#rename()->renames pandas dataframe columns
df=pd.DataFrame(tech,index=row_nam)
#assign new header by setting new column name
df.columns=['A','B','C','D']

#rename column names using rename() method
df=pd.DataFrame(tech,index=row_nam)
df.columns=['A','B','C','D']
df2=df.rename({'A':'c1','B':'c2'},axis=1)
df2=df.rename({'C':'c3','D':'c4'},axis='columns')
df2=df.rename(columns={'A':'c1','B':'c2'})

#Drop Dataframe Rows and Columns
df=pd.DataFrame(tech,index=row_nam)
#drop by row labels
df1=df.drop(['a','b'])
#drop by rows by positon/index
df1=df.drop(df.index[1])
df1=df.drop(df.index[[1,3]])
#drop by index range
df1=df.drop(df.index[2:])

#when you have default indexes for rows
df=pd.DataFrame(tech)
df1=df.drop(0)
df=pd.DataFrame(tech)
df1=df.drop([0,3],axis=0)#it will delete row0 and row3
df1=df.drop(range(0,3))#it will delete 0 and 1
#drop column by name
#drop from column
d21=df.drop(df.index[2:])


###################10/04/24######################################
#explicitly using parameters name "labels"
df2=df.drop(labels=['fees'],axis=1)

#Alternatively you can also use columns instead of labels
df2=df.drop(columns=['fees'],axis=1)

#drop column by index
print(df.drop(df.columns[1],axis=1))
df=pd.DataFrame(tech)

#using inplace True
df.drop(df.columns[2],axis=1,inplace=True)
print(df)

#drop two or more columns by label name
df=pd.DataFrame(tech)
df2=df.drop(columns=['fees','courses'],axis=1)

#drop two or more columns by index
df=pd.DataFrame(tech)
df2=df.drop(df.columns[[0,1]],axis=1)
print(df2)
###################################################
#drop columns from list of columns                #
df=pd.DataFrame(tech)                             #
df.columns                                        #
listcol=['courses','fees','duration','discount']  #
df2=df.drop(listcol,axis=1)                       #
print(df2)                                        #
###################################################

#remove columns from dataframe inplace
df=pd.DataFrame(tech)                             
df.drop(df.columns[1],axis=1,inplace=True)
df
#using inplace=True


#################################################################
#Pandas select rows by index(position/label)use of iloc & loc
df=pd.DataFrame(tech)     
#iloc=access columns/rows by index                        
#syntax->df.iloc[startrow:endrow,startcolumn:endcolumn]
df2=df.iloc[0:2,0:2]

#for all rows/columns use [:]
df2=df.iloc[:,:]#all rows & columns

df2=df.iloc[:,0:2]#all rows two columns
#The first slice [:] indicates to return all rows
#the second slice specifies that only columns
#between 0 and 2 (excluding 2) should be returned

df2=df.iloc[0:2,:]#two rows all columns
df2
#in this case , the first slice [0:2] is requesting only row 0
#through lof the dataframe. 
#the second slice [:] indicates that all columns are required

#slicing Specific rowand Columns using iloc attribute
df3=df.iloc[1:2,1:3]
df3

#select rows by integer index
df2=df.iloc[2]#select row by index
df2
df2=df.iloc[[2,3,4]] #select rows by index list
df2=df.iloc[1:5]     #select rows by integer index range
df2=df.iloc[:1]      #select first row
df2=df.iloc[:3]      #select first 3 row
df2=df.iloc[-1:]     #select last row
df2=df.iloc[-3:]     #select last 3 row
df2=df.iloc[::2]     #select alternate row



'''                       12/04/24                    '''
#Select rows by index labels
df2=df.loc['b']     #select row by row-label
df2=df.loc[['a','c','e']]#select rows by row-labels
df2=df.loc['c':'e'] #select rows by label range
df2=df.loc['a':'e':2] #select alternate rows by label range


#Accessing columns by columns name or index
#by using df[] notation
df2=df['fees']
#Select multiple columns
df2=df[['courses','fees']]


#using loc[] to take column slices
#loc[] syntax to slice columns->df.loc[:,start:stop:step]
#Select random columns/multiple columns
df2=df.loc[:,['courses','fees','duration']]
#Select columns between two columns
df2=df.loc[:,'fees':'discount']
#Select columns by range
df2=df.loc[:,'duration':]
#All the columns upto 'duration'
df2=df.loc[:,:'duration']
#Select every alternate row
df2=df.loc[:,::2]     


##Pandas.DataFrame.query() by examples
#Query all rows with courses equals 'spark'
df2=df.query("courses=='spark'")
print(df2)
df2=df.query("courses!='spark'")
print(df2)

####Pandas add column to dataframe
tutors=['rehan','ram','roy','rahul','rocky']
df2=df.assign(tutorassigned=tutors)

#Add multiple columns to dataframe
Mnccompany=['google','microsoft','infosys','x','amazon']
df2=df.assign(MNC=Mnccompany,tutors=tutors)
df2

#We can derive new column from existing column
df2=df.assign(dis_per=lambda x:x.fees*x.discount/100)
df2

#Append column to existing pandas dataframe
#Add new columns to the existing dataframe
df['MNCS']=Mnccompany
df

#Adding a new column to a specific location
df.insert(0,"Tutors", tutors)
df

#Renaming columns 
#rename a single column
df2=df.rename(columns={'courses':'courses_list'})
print(df2.columns)
#rename multiple column
df2=df.rename(columns={'fees':'paisa','courses':'courses_list'})
#rename a single column in orignal dataframe use inplace
df.rename(columns={'fees':'paisa','courses':'courses_list'},inplace=True)


'''                    15/04/24                        '''
#quick ex of get the number of rows in DataFrame
r_count=len(df.index)
r_count
r_count=len(df.axes[0])
r_count
c_count=len(df.axes[0])
c_count

r_count=df.shape[0]#returns no of rows
r_count
c_count=df.shape[1]#returns no of columns
c_count


#Pandas apply function to a column 
#Using Dataframe.apply() to apply function add column
import pandas as pd
import numpy as np
data={"A":[1,2,3],
      "B":[4,5,6],
      "C":[7,8,9]}
df=pd.DataFrame(data)
print(df)
def add_3(x):
    return x+3
df2=df.apply(add_3)
df2

df2=(df.A).apply(add_3)#for column A only

###Using apply function single column
def add_3(x):
    return x+3
df["B"]=df["B"].apply(add_3)
df["B"]
#Apply function tu multiple columns
df[["A","B"]]=df[["A","B"]].apply(add_3)
df

#Apply a lambda function to each column
df["A"]=df["A"].apply(lambda x:x-2)
print(df)

#Using pandas DataFrame.transform() to apply function col
#using DataFrame.transform()
def add_2(x):
    return x+2
df2=df.transform(add_2)
df2

#Using pandas DataFrame.map()
df["B"]=df["B"].map(lambda B:B/2)
df

#using numpy function on single column
#using Dataframe.apply() & []opeator
df=pd.DataFrame(data)
import numpy as np
df['A']=df['A'].apply(np.square)
print(df)

#Using NumPy.Square() method
#using numpy.square & []opeator
df['A']=np.square(df['A'])
print(df)

#pandas groupby() with examples
import numpy as np
import pandas as pd
tech={
      'courses':['pandas','pyspark','pyspark','hadoop','pandas'],
      'fees':[20000,3000,np.nan,5000,6000],
      'duration':['30days','','50days','60days','70days'],
      'discount':[1.2,2.3,3.4,4.5,5.6]
      }
row_nam=['r0','r1','r2','r3','r4']
df=pd.DataFrame(tech,index=row_nam)
print(df)
#on single column
df2=df.groupby(['courses']).sum()
df2
#on multiple column
df2=df.groupby(['courses','duration']).sum()
df2

#Add index to the grouped data
#add row index By group by result
df2=df.groupby(['courses','duration']).sum().reset_index()
df2

#Get the list of all columns names from headers
column_headers=list(df.columns.values)
print('The column header:',column_headers)

#Using list(df) to get the columns as a list
column_headers=list(df.columns)
print('The column header:',column_headers)


'''                   16/04/24                         '''
#Pandas shuffle DataFrame Rows
#shuffle the DataFrame rows and return all rows
df1=df.sample(frac=1)
print(df1)

#create new index starting from 0
df1=df.sample(frac=1).reset_index()
print(df1)

#Drop shuffle index
df1=df.sample(frac=1).reset_index(drop=True)
print(df1)

###create 2 dataframes
import numpy as np
import pandas as pd
tech1={
      'courses':['spark','pyspark','python','pandas'],
      'fees':[20000,25000,22000,30000],
      'duration':['30days','40days','35days','50days']
      }
row_nam=['r1','r2','r3','r4']
df1=pd.DataFrame(tech1,index=row_nam)
print(df1)

import numpy as np
import pandas as pd
tech2={
      'courses':['spark','java','python','go'],
      'fees':[2000,2300,1200,2000]
      }
row_nam=['r1','r6','r3','r5']
df2=pd.DataFrame(tech2,index=row_nam)
print(df2)

#Pandas join#if no mention it is left join
df3=df1.join(df2,lsuffix="_Left",rsuffix="r_right")
print(df3)

#pandas Inner join
df3=df1.join(df2,lsuffix="_Left",rsuffix="r_right",how="inner")
print(df3)

#pandas right join
df3=df1.join(df2,lsuffix="_Left",rsuffix="r_right",how="right")
print(df3)

#pandas left join
df3=df1.join(df2,lsuffix="_Left",rsuffix="r_right",how="left")
print(df3)

#Using pandas.merge()
df3=pd.merge(df1, df2) 
print(df3)
#Using dataframe.merge()
df3=df1.merge(df2)
print(df3)

#Using pandas.concat() to concat two dataframes
data=[df1,df2]
df2=pd.concat(data)
df2
#horizontaly concate
df3=pd.concat(data,axis=1).reset_index()
df3

#Concating two or more Dataframes
df1=pd.DataFrame({"Course":["Spark ","Pyspark","Python ","PAndas"],
       "Fee":[20000,25000,22000,30000]})
df2=pd.DataFrame({"Course":["Unix","Hadoop","Hyperion","Java"],
       "Fee":[20000,25000,22000,30000]})
df=pd.DataFrame({"Duration":["30days","40days","35days","60days","55days"],
                 "Discount":[3000,2300,2500,2000,3000]})
data=([df1,df2,df])
df5=pd.concat(data)


'''                     18/04/24                   '''
#wrtie dataframe to excel file
df.to_csv('c:/1-python/courses.xlsx')

df=pd.read_excel('c:/1-python/courses.xlsx')
print(df)


####using series.values.tolist()
col_list=df.courses.values.tolist()
print(col_list)

col_list=df['courses'].values.tolist()
print(col_list)

#Using list function
col_list=list[df["courses"]]
print(col_list)

#convert to numpy array
col_list=df['courses'].to_numpy()
print(col_list)

#Arrays in NumPy
#Create ndarray
import numpy as np
arr=np.array([10,20,30])
print(arr)

#Create multidimensional array
arr=np.array([[10,20,30],[40,50,60]])
print(arr)

#Use ndmin parameter to specify how many minimum D you want
#to create an array with minimum dimensions
arr=np.array([10,20,30],ndmin=3)
print(arr)


#To change the data type dtype parameter
arr=np.array([10,20,30],dtype=complex)
print(arr)

#To get the Dimensions of array
arr=np.array([[10,20,30],[40,50,60]])
print(arr.ndim)
print(arr)


#Get datatype of array
arr=np.array([[10,20,30],[40,50,60]])
print(arr.dtype)

#Get shape and size of array
arr=np.array([[10,20,30],[40,50,60]])
print(arr.size)#gives no of element
print(arr.shape)

#convert int to float
arr=np.array([[10,20,30],[40,50,60]],dtype=float)
print(arr)

#Create a sequence of integers  using arange()
#create a sequence of integers from 0 t o 20 with step 3
arr=np.arange(0,20,3)
print("A sequence array with step of 3:\n",arr)

#Array indexing in NumPy
#Access single element using index
arr=np.arange(11)
print(arr)
print(arr[2])
print(arr[-2])

#Accessing multi-D  element using index
arr=np.array([[5,20,35,50],[50,65,80,95]])
print(arr.shape)
print(arr)
print(arr[0,1])
print(arr[1,-1])


##Accessing array element using slicing
arr=np.array([0,1,2,3,4,5,6,7,8,9])
x=arr[0:8:2] #[start:stop:step]
x
x=arr[-2:3:-1]
x

#Accessing multi-D  element using slicing
arr=np.array([[10,20,10,40],[40,50,70,90],
              [60,10,70,80],[30,90,40,30]])
print(arr)
print(arr[1,2])
print(arr[1,:])#all values of row 1
print(arr[:,2])#all values of column 2
print(arr[:3,::2])#rows 0,1,2 & alternate columns 



'''                    19/04/24                              '''
#Integer array indexing
arr=np.arange(20).reshape(5, 4)
print(arr)

#Boolean Array indexing
#This is advanced indexing occurs when an obj is an array of
arr=np.arange(12).reshape(3, 4)
print(arr)
rows=np.array([False,True,True])
rows
wanted_rows=arr[rows,:]
print(wanted_rows)

#numpy asarray
list=[20,30,40,50]
array=np.array(list)
print(array)
print(type(array))

#Numpy Array properties
#ndarray.shape
#ndarray.ndim
#ndarray.itemsize
#ndarray.size
#ndarray.dtype

#Shape
arr=np.array([[10,20,10],[40,50,70]])
print(arr.shape)

arr=np.array([[10,20,10],[40,50,70]])
arr.shape=(3,2)
print(arr)

#reshape usage
arr=np.array([[10,20,10],[40,50,70]])
new_arr=arr.reshape(3,2)
print(new_arr)

#Arithmatic Operations
arr1=np.arange(16).reshape(4,4)
arr2=np.array([1,2,3,4])
#add()
add_arr=np.add(arr1,arr2)
print(f"Adding two arrays:\n{add_arr}")
#subtract()
sub_arr=np.subtract(arr1,arr2)
print(f"Subtracting two arrays:\n{sub_arr}")
#multiply()
mul_arr=np.multiply(arr1,arr2)
print(f"Multiplying two arrays:\n{mul_arr}")
#divide()
div_arr=np.divide(arr1,arr2)
print(f"Dividing two arrays:\n{div_arr}")

#To perform Reciprocal operation
arr1=np.array([50,10.3,5,1,200])
recp_arr=np.reciprocal(arr1)
print(f"Reciprocal array:\n{recp_arr}")


#numpy.power()
arr1=np.array([3,5,20])
pow_arr1=np.power(arr1,2)
print(f"Powering array:\n{pow_arr1}")


arr1=np.array([3,5,20])
arr2=np.array([1,2,3])
pow_arr2=np.power(arr1,arr2)
#element of arr1 will be powered to element of arr2
print(f"Powering array:\n{pow_arr2}")

#numpy.mod()
arr1=np.array([7,20,13])
arr2=np.array([3,5,2])
mod_arr=np.mod(arr1,arr2)
print(f"Array after applying mod:\n{mod_arr}")

#Create an empty array
from numpy import empty
a=empty([2,2])
print(a)

#Create an Zero array
from numpy import zeros
a=zeros([2,2])
print(a)

#Create one array
from numpy import ones
a=ones([2,2])
print(a)

#Create array with vstack
from numpy import array
from numpy import vstack
a1=array([1,2,3])
a2=array([4,5,6])
a3=vstack((a1,a2))
print(a3)
print(a3.shape)


#Create array with hstack
from numpy import array
from numpy import hstack
a1=array([1,2,3])
a2=array([4,5,6])
a3=hstack((a1,a2))
print(a3)
print(a3.shape)

'''                          23/04/24                    '''
#index row of 2D array
from numpy import array
data=array([[11,22],[44,33],[88,55],[66,77]])
print(data[0,])#0th row and all column

#Negative slicing
from numpy import array
data=array([11,22,44,33,88,55,66,77])
print(data[-2:])#last two elements

#Split input and output data
data=array([
    [11,22,13],
    [44,33,43],
    [88,55,99],
    [66,77,67]])
#Separate data
x,y=data[:,:-1],data[:,-1]#last column element
x
y

#broadcast scalar to one D array
a=array([11,22,44,33,88,55,66,77])
print(a)
#define scalar
b=2
print(b)
#broadcast
c=a+b
print(c)


'''Vector L1 norm
The L1 norm is calculated as the sum of the absolute vector
values,where the absolute value of the scalar uses the notation
|a1|.
In effect,the norm is a collection ot the Manhattan distance
from the origin of the vector space.
||v||1=|a1|+|a2|+|a3|''' 

from numpy import array
from numpy.linalg import norm
a=array([1,2,3,4,5,6,7,8,9,10])
print(a)
#calculate norm
l1=norm(a,1)#Sum of all
print(l1)

'''The notation for the L2 norm of a vector x is ||x||power of 2.
To calculate the L2 norm of a vector, take the square root of the 
sum of the squared vector values.
Another name for L2 norm of a vector is Euclidean distance.
This is often used for calculating the error in 
machine learning models.'''
from numpy import array
from numpy.linalg import norm
a=array([1,2,3])
print(a)
#calculate norm
l2=norm(a)#1+4+9=14 under root=3.7
print(l2)


#Triangular matrices
from numpy import array
from numpy import tril
from numpy import triu
#define square matrix
M=array([
    [1,2,3],
    [1,2,3],
    [1,2,3]])
print(M)
#lower triangular matrix
lower=tril(M)
print(lower)
upper=triu(M)
print(upper)


#Diagonal matrix
from numpy import array
from numpy import diag
M=array([
    [1,2,3],
    [1,0,3],
    [1,2,3]])
print(M)
d=diag(M)
print(d)
D=diag(d)
print(D)


#Identity matrix
from numpy import identity
i=identity(5)
print(i)


#Orthogonal matrix
'''The matrix is said to be orthogaonal matrix if the product of 
a matrix and its transpose gives an identity value'''
from numpy import array
from numpy.linalg import inv
Q=array([[1,0],
         [0,1]])
print(Q)
#Inverse equvalence
V=inv(Q)
print(Q.T)
print(V)
#Identity equivalence
I=Q.dot(Q.T)
print(I)


'''                          24/04/24                    '''
#calculate Transpose
from numpy import array
A=array([
    [1,2],
    [3,4],
    [5,6]])
print(A)
C=A.T
print(C)

#inverse matrix
from numpy import array
from numpy.linalg import inv
A=array([
    [1,2],
    [3,4]])
print(A)
B=inv(A)
print(B)

#Multiply A and B
I=A.dot(B)
print(I)

##Sparse Matrix
from numpy import array
from scipy.sparse import csr_matrix
#create a dense matrix
A=array([
    [1,0,0,1,0,0],
    [0,0,2,0,0,1],
    [0,0,0,2,0,0]])
print(A)
#convert to sparse matrix(csr method)
s=csr_matrix(A)
print(s)
#reconstruct to dense
B=s.todense()
print(B)






