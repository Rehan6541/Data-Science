###11)26/03/24

#read csv file
import pandas as pd
f1=pd.read_csv('C:/1-Python/buzzers.csv')
f1

#check for the working directory
import os
with open('C:/1-Python/buzzers.csv') as raw_data:
    print(raw_data.read())

#reading csv file as a list
import csv
with open('C:/1-Python/buzzers.csv') as raw_data:
    for line in csv.reader(raw_data):
        print(line)
    
#reading csv file as a dictionary
import csv
with open('C:/1-Python/buzzers.csv') as raw_data:
    for line in csv.DictReader(raw_data):
        print(line)

##############################################################
###ERROR
with open('C:/1-Python/buzzers.csv') as raw_data:
    #ignore=data.readline()
    flights={}
    for line in raw_data:
        k,v=line.split(',')
        flights[k]=v
flights        
    

##pre requisite of decorators
def plus_one(num):
    num1=num+1        
    return num1    
plus_one(5)

#defining functions inside other functions
def plus_one(num):
    def add_one(num):
        num1=num+1        
        return num1
    result=add_one(num)#thiswill notgive numit willcreate obj.
    return result
plus_one(5)

##passing function as argument to other function
def plus_one(num):
    result1=num+1        
    return result1  

def function_call(function):
    result=function(5)
    return result
function_call(plus_one)

##functions returning other functions
def hello_fun():
    def say_hi():
        return "Hi"
    return say_hi()
hello=hello_fun()
hello

#Need for decorators
import time
#num_list=[1,2,3]
def calc_sq(numbers):
    start=time.time()
    result=[]
    for number in numbers:
        result.append(number*number)
    end=time.time()
    total_time=(end-start)*1000
    print(f"Total time for execution square is {total_time}")
    return result
#calc_sq(num_list)


def calc_cube(numbers):
    start=time.time()
    result=[]
    for number in numbers:
        result.append(number*number*number)
    end=time.time()
    total_time=(end-start)*1000
    print(f"Total time for execution cube is {total_time}")
    return result
#calc_cube(num_list)

array=range(1,1000000)
out_sq=calc_sq(array)
out_cube=calc_cube(array)



##############27/03/24##############


'''A python decorator is a function that takes in a function
and returns it by adding some functionality'''
def say_hi():
    return "hello moto"
def upper_case(function):
    def wrapper():
        func=function()
        make_upper=func.upper()
        return make_upper
    return wrapper
decorate=upper_case(say_hi)
decorate()

'''however python provides easier way to use decorators.
we simple use the @ symbol before the function we would
like to decorate'''

def upper_case(function):
    def wrapper():
        func=function()
        make_upper=func.upper()
        return make_upper
    return wrapper
@upper_case
def say_hi():
    return "hello moto"
say_hi()

'''we can use multiple decorators to a single function
however,the decorators will be applied in the order that
we have called them'''



def split_str(function):
    def wrapper():
        func=function()
        splited_str=func.split()
        return splited_str 
    return wrapper
def upper_case(function):
    def wrapper():
        func=function()
        make_upper=func.upper()
        return make_upper
    return wrapper
@split_str
@upper_case
def say_hi():
    return "hello moto"
say_hi()


#time calculation program
import time
def time_it(func):
    def wrapper(*args,**kwargs):
        start=time.time()
        result=func(*args,**kwargs)
        end=time.time()
        total_time=(end-start)*1000
        print(func.__name__+f"took {total_time} mili sec")
        return result
    return wrapper
@time_it
def calc_sq(numbers):
    result=[]
    for number in numbers:
        result.append(number*number)
    return result

@time_it
def calc_cube(numbers):
    result=[]
    for number in numbers:
        result.append(number*number*number)
    return result

array=range(1,1000000)
out_sq=calc_sq(array)
out_cube=calc_cube(array)


#########28/03/24
###########Exception Handling
'''Sometimes a single piece of code might be suspectedd to have
 more then one tye of error for handling such situation we 
 can have multiple except blocks for a single try block'''

try:
    num=50
    denom=int(input("Enter the denom:"))
    print(num/denom)
    print("Div performed successfully")
except ZeroDivisionError:
    print("Denom as zero is not allowed")
except ValueError:
    print("Only integers should be entered")
    
    
try:
    num=50
    denom=int(input("Enter the denom:"))
    print(num/denom)
    print("Div performed successfully")
except ValueError:
    print("Only integers should be entered")
except :
    print("OOPS....some exception raised")  
    
    
###Exception handling with try...except and else
try:
    num=50
    denom=int(input("Enter the denom:"))
    quo=num/denom
    print("Div performed successfully")
except ZeroDivisionError:
    print("Denom as zero is not allowed")
except ValueError:
    print("Only integers should be entered")
else:
    print("The result is",quo)

###Exception handling with try...except..else and finally
try:
    num=50
    denom=int(input("Enter the denom:"))
    quo=num/denom
    print("Div performed successfully")
except ZeroDivisionError:
    print("Denom as zero is not allowed")
except ValueError:
    print("Only integers should be entered")
else:
    print("The result is",quo)
finally:
    print("Over and out")
        
 
  




    
    
    
    
    
    
    
    
    