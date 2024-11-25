#Q)10
Define a array data = array([11, 22, 33, 44, 55]) and slice it from 1 to 4
->
import numpy as np
data=np.array([11, 22, 33, 44, 55])
print(data[1:5])

#Q)9
Write a Pandas program to create the specified columns and rows from a given data frame.
name: ['Anna', 'Dinu', Ramu', 'Ganu', 'Emily', 'Mahesh', 'Jayesh', ‘venkat', 'Ajay', 'Dhanesh']
'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19]
'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1]
'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
->
import pandas as pd
df={'name': ['Anna', 'Dinu', 'Ramu', 'Ganu', 'Emily', 'Mahesh', 'Jayesh', 'venkat', 'Ajay', 'Dhanesh'],
'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df1=pd.DataFrame(df,index=labels)

#Q)8
Write a Python program to filter a list of integers using Lambda. 
Original list of integers:
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Even numbers from the said list:
[2, 4, 6, 8, 10]
Odd numbers from the said list:
[1, 3, 5, 7, 9]
->
or_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even=list(filter(lambda x:(x%2 ==0),or_list))
print(even)  
odd=list(filter(lambda x:(x%2 !=0),or_list))
print(odd)        
    
#Q)7 
Define a array ,data = array([11, 22, 33, 44, 55]) find 0 th index 4 th index data 
->
import numpy as np  
data =np.array([11, 22, 33, 44, 55])   
print(data[0])              
print(data[4])                
              
#Q)6              
Open the file data.txt using file operations
 ->    
import pandas as pd
a=pd.read_csv("data.txt")         
print(a)

#Q)5
Using dict comprehension and a conditional argument create a dictionary from the current dictionary 
where only the key:value pairs with value above 2000 are taken to the new dictionary              
->
dict={"a":1000,"b":5000,"c":10000,"d":12345,"e":123}            
print(dict)  
for k,v in dict:
    print(k,v)
  
#Q)4
Write a Python program to iterate over dictionaries using for loops
->
dict={"a":1000,"b":5000,"c":10000,"d":12345,"e":123}            
print(dict)  
for value in dict:
    print(value)



#Q)3
Write a Python program to reverse a string         
->
str="REHAN"
rev=str[::-1]
print(rev)

#Q)2
Use list comprehension to construct a new list but add 6 to each item.
->
list1=[1,2,3,4,5]
list6=[]
for x in list1:
    x=x+6
    list6.append(x)
print(list6)

list1=[1,2,3,4,5]
list6=for x in list1 lambda x:x+6
print(list6)


#Q1
Write a Python function that takes two lists and returns True if they have at least one common 
member.
a=[1,2,3,4,5]
b=[6,7,8,9,5]

def common(a,b):
    for x in a:
        for y in b:
            if x==b:
                print("abc")
common(a,b)   
print(common(a, b))    
    
    
    