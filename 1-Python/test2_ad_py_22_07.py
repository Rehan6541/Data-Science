#Mon Jul 22 09:04:26 2024
#Test 2 Advance Python

1. Check if email address valid or not in Python
e.g. Input: my.ownsite@ourearth.org
Output: Valid Email
Input: ankitrai326.com
Output: Invalid Email

import re
Email=input("Enter your email:")
pattern='[a-zA-Z0-9_]*@[a-z]*\.[a-zA-Z0-9]*'
matches=re.findall(pattern,Email)
if matches:
    print("valid email")
else:
    print("invalid email")
   

2. Write a Python program to find the median of below three values.
Values: (25,55,65)

values=[25, 55, 65]
values.sort()
mid=len(values)//2
median=values[mid]
print(f"The median of {values} is {median}")


3. Write a program to create a decorator function to measure the
execution time of a function.

def decorator():
    def
    
    
    


4. Write a python program that opens a file and handles a
FileNotFoundError exception if the file does not exist.

import pandas as pd
file=input("Enter filename")
try:
    a=pd.read_csv(file)
except FileNotFoundError:
    print(f"Error: The file '{file}' does not exist.")
   

try:
    file = open("file_name.txt", "r")
    file.close()
except FileNotFoundError:
        print("File not found error")        
        
5. Write a python program to find the intersection of two given arrays
using Lambda.
Original arrays:
[1, 2, 3, 5, 7, 8, 9, 10]
[1, 2, 4, 8, 9]
Intersection of the said arrays: [1, 2, 8, 9]

a=[1, 2, 3, 5, 7, 8, 9, 10]
b=[1, 2, 4, 8, 9]
common=list(filter(lambda x:x in a,b))
print("Intersection of the said arrays:",common)










