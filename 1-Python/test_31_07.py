#Created on Wed Jul 31 09:07:20 2024

#Test
1. Write a Pandas program to convert Series of lists to one Series.
Sample Output: 
Original Series of list
0 [Red, Green, White]
1 [Red, Black]
2 [Yellow]
dtype: object
One Series
0 Red
1 Green
2 White
3 Red
4 Black
5 Yellow
dtype: object

import pandas as pd
df=pd.Series(['Red', 'Green', 'White'])
df1=pd.Series(['Red', 'Black'])
df2=pd.Series(['Yellow'])
df3=pd.Series([['Red', 'Green', 'White'],['Red', 'Black'],['Yellow']])
df3
df4=pd.Series(['Red', 'Green', 'White','Red','Black','Yellow'])
df4

2. Write a python NLTK program to split the text sentence/paragraph into 
a list of words.
text = '''
Joe waited for the train. The train was late. 
Mary and Samantha took the bus. 
I looked for Mary and Samantha at the bus station.
'''
from nltk import word_tokenize
a=word_tokenize(text)
a
b=text.split()
b

3. Create a result array by adding the following two NumPy arrays. Next, 
modify the result array by calculating the square of each element
import numpy
arrayOne=numpy.array([[5, 6, 9], [21 ,18, 27]])
arrayTwo=numpy.array([[15 ,33, 24], [4 ,7, 1]])
arraythree=(arrayOne+arrayTwo)
arraythree
arraysquare=arraythree**2
arraysquare

4. Write a python program to extract word mention someone in tweets 
using @ from the specified column of a given DataFrame.
DataFrame: ({'tweets': ['@Obama says goodbye','Retweets for @cash','A political endorsement in @Indonesia', '1 dog = many #retweets', 'Just a simple #egg']})
import re
pattern='@[a-z]*\.[a-zA-Z0-9]'
match=re.
            
            
5. Write a NumPy program to compute the mean, standard deviation, and 
variance of a given array along the second axis.
import numpy as np
df=np.array([0,1,2,3,8,5]) 
mean=df.mean()         
mean
var=df.var()
var
sd=df.std()
sd

