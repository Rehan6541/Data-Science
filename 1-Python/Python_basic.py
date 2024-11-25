#############################

'''print("Hello World")
x=1
print(x)
print(type(x))
x=10000.0
print(x)
print(type(x))

age=input("Enter the age")
print(type(age))
print(age)
age1=input("Enter the age1")
print(type(age1))
print(age1)
print(age+age1)

#by default input data type is string

age2=int(input("Enter the age1"))
print(type(age2))
print(age2)
age3=int(input("Enter the age1"))
print(type(age3))
print(age3)
print(age2+age3)'''

###############################
#int,string to float
'''a=10#int
c="20.05"#string
print(a)
print(type(a))
print(c)
print(type(c))
b=float(a)#float
print("int to float",b)
print(type(b))
d=float(c)#float
print(type(d))
print("string to float",d)'''

###############################

#complex no
'''c=1
c1=2j
print("c:",c,",c1:",c1)
print(type(c))
print(type(c1))
print(c.real)
print(c1.imag)'''

###############################

#boolean values(True or False)
'''all_ok=True
print(all_ok)
all_ok=False
print(all_ok)
print(type(all_ok))

status=bool(input("Ok it is confirmed?:"))
print(status)
print(type(status))'''

###############################

#arithmatic operations
h=5
a=20
print(a+h)
print(type(a+h))
print(a-h)
print(type(a-h))
print(a*h)
print(type(a*h))
print(a/h)
print(type(a/h))

#integer division operator=//
print(100//20)
print(type(100//20))

#modolu operator for remainder=%
print("modulu division 100%13:",100%13)
print("modulu division 3%2:",3%2)

#power operator**
a=5
b=3
print(a**b)

#assignment operator=,+=,-=,*= etc
#comparision operator==
x=0
x+=1

#None operator
winner=None
print(winner)
print(winner is None)
print(winner is not None)

#if and comparator operator
num=int(input("Enter the value:"))
if num>0:
    print(num)
   
#else in if statement    
num1=int(input("Enter the value:")) 
if num1<0:
    print(num1,"is negative")
else:
    print(num1,"is positive")
    
#elif multiple condition
saving=int(input("Enter the savings:")) 
if saving==0:
    print(saving,"is zero")
elif saving<500:
    print(saving,"is good")
elif saving<1000:
    print(saving,"is very good")
elif saving<10000:
    print(saving,"is very very very good")
else:
     print("Byeeee")   
  
#while loop
count=1
print("stating....")
while count<=10:
    print(count)
    count+=1

#for loop
print("print out values in range")
for i in range(2,10):
    print(i)
print("done")   

print("only print code if all iterations complete")
num=int(input("Enter the value:"))
for i in range(0,16):
    if i==num:
        break
    print(i)
print("done")    

#anonymous _ variable
for _ in range(0,10):
    print("$",end="")
    print("") 


#########################################
# '''5th Session 20/03'''

#odd numbers in range
start ,end=4,19
for num in range(start,end+1):
    if num%2!=0:
        print(num,end=" ")
        

#even numbers in range
start,end=4,22
for num in range(start,end+1):
    if num%2==0:
        print(num)
        
#variables       
x,y,z=5,6,7
print(x)
print(y)
print(z)
x,y,z=5
print(x)
print(y)
print(z)


#global variable
x="awesome"
def my_function():
    print("python is "+x)
my_function()


#global variable and local variable
x="awesome"
def my_function():
    x="funtastic"
    print("python is "+x)
my_function()
print("python is "+x)


#getting data types
x=5
print(type(x))
x=range(6)
print(x)
print(type(x))
x={"name":"Ishwar","age":"20"}
print(x)
print(type(x))


#string concate
str1="Hello"
str2=2
print(str1+str2)#error due to diff datatype
str1="Hello"
str2=str(2)
print(str1+str2)


#if you want muptiple strings
x="""This is python .It is very Powerful"""
print(x)
print(x[2:8])
print(x[:3])   #slicing from the start
print(x[4:])   #slicing to the end
print(x[-5:-2]) #negative indexing
print(x.upper())
print(x.lower())


#remove white spaace ,removes white spaace from intitial part of string
x="  this is python"
print(x.strip())

#replace the string
x="Hello world"
print(x.replace("Hello","Namaste"))

#use of split which replaces space
x="Hello world"
print(x.split(" "))

x="Hello world"
string1=x[::-2]
print(string1)

x="Hello world"
string1=x[::-1]#string reversing
print(string1)

#find method,searches the string for a specified value at
x="This is python and it is very powerful"
print(x.find("and"))

#string concatness
x="Hello"
y="World"
print(x+y)
print(x+" "+y)

#string format
x=20
u="My name is Rehan"
print(x+y)#error
print(f"My name is Rehan and my age is {x}")

quantity=3
item=2
price=30
print(f"I want {quantity} pieces and item number is {item},its price is {price}")
#another method
myorder="I want {} and item number is {} its price is {}"
print(myorder.format(quantity,item,price))
#another method
quantity=3
item=2
price=30
myorder="I want {0} and item number is {1} its price is {2}"
print(myorder.format(quantity,item,price))

#the escape character allows you to use double quotes
text="This is a fun fair and it has got big \"round rigo\""
print(text)

#operator precedence PEMDAS RULE
'''P=Parenthesis
E=Exponential
M=Multiplication
D=Division
A=Addition
S=Substraction'''
a=3+3*3/3-3
print(a)


###LIST[]###

list=["apple","mango","cherry"]
#       0        1       2
print(list)
#list items are indexed,the first index is [0]
print(list[0])
print(list[2])

#append()->adds an item in the list at the end
list=["apple","mango","cherry"]
list.append("banana")
list=["apple","mango","cherry"]

#clear()-> removes all the item of the list
list=["apple","mango","cherry"]
list.clear()
print(list)

#copy()-> used to copy list
list=["apple","mango","cherry"]
list2=list.copy()
print(list2)

#count()->used to count the elements
list=["apple","mango","apple","cherry"]
print(list.count("apple"))

#extend()->join the elements of two list
list=[1,2,3]
list1=[4,5,6]
list.extend(list1)
print(list)

#insert()->insert the element in the list at required position
list=["apple","mango","cherry"]
list.insert(1,"banana")
print(list)

#pop()->Removes the element at a specific position
list=["apple","mango","cherry"]
list.pop(1)
print(list)

#remove()->Removes the element of a specific name
list=["apple","mango","cherry"]
list.remove("apple")
print(list)

#reverse()->Reverse the elements of list
list=["apple","mango","cherry"]
list.reverse()
print(list)

#sort()->sort the list aplhabatically and in ascending order
list1=[1,43,3]
list=["apple","mango","cherry"]
list.sort()
print(list)
list1.sort()
print(list1)


###TUPLE()###

tup=("apple","mango","cherry")
#       0       1        2
print(tup)
print(tup[2])

#once it is created,you cannot change it,to change convert it into list modify again convert it into tuple
tup=("apple","mango","cherry")
#tup[1]="kiwi"#error 
#first convert it into list
y=list(tup)
y[1]="kiwi"
x=tuple(y)
print(y)

#you can access the elements in tuple by indexes
tup=("apple","mango","cherry")
print(tup[1])

#to join two tuples you can use + operator
tup=("apple","mango","cherry")
tup1=(1,2,3)
tupp=tup+tup1
print(tupp)


###DICTIONARY{key:value}###

dic={"Brand":"Maruti","Model":1123,"Year":2020,1:5}
print(dic)
print(len(dic))#length of dictionary
dic1={"Brand":["Maruti","Mahindra","Toyata"],"Model":[1123,3456,7689],"Year":[2020,2021,"two"]}
print(dic1)
print(len(dic1))
print(dic1.get("Model"))#gives values of key model
print(dic1.keys())#prints all the keys in the dic

####6)21/03/24####

car={"Brand":"Ford","Model":"Mustand","Year":2020}
x=car.keys()
print(x)
car["Color"]="White"#to add a key value pair in dict
x=car.keys()
print(x)

#pop()->removes the dicitionary requires keys
car={"Brand":"Ford","Model":"Mustand","Year":2020}
car.pop("Model")
print(car)

#accessing keys in the dicitionary
car={"Brand":"Ford","Model":"Mustand","Year":2020}
for x in car:
    print(x)
    
#accessing values in the dicitionary
car={"Brand":"Ford","Model":"Mustand","Year":2020}
for x in car:
    print(car[x])  
   
#accessing keys & values in the dicitionary
car={"Brand":"Ford","Model":"Mustand","Year":2020}
for key,value in car.items():
    print("%s == %s" % (key,value))

#copying the dict
car={"Brand":"Ford","Model":"Mustand","Year":2020}
car1=car.copy()
print(car1)
#another way of copying
car2=dict(car) 
print(car2)

#a dict can contain dicts
#this is called nested dict
our_family={"child1":{"name":"Ram","age":20},
            "child2":{"name":"Sham","age":19}}
print(our_family)

#dict methods

#clear()->removes all the elements in the dict
car={"Brand":"Ford","Model":"Mustand","Year":2020}
car.clear()
print(car)

#fromkeys()->create a dict with keys all values
x={"key1","key2","key3"}
y=0
z=dict.fromkeys(x,y)
print(z)

#get()->to get the values of dict
car={"Brand":"Ford","Model":"Mustand","Year":2020}
print(car.get("Brand"))

#items()->returns the dict's key-value
car={"Brand":"Ford","Model":"Mustand","Year":2020}
print(car.items())

#values()->display all the values in dict
car={"Brand":"Ford","Model":"Mustand","Year":2020}
print(car.values())

#update()->insert an item in dict
car={"Brand":"Ford","Model":"Mustand","Year":2020}
car.update({"color":"black"})
print(car)


#for loop
list=["apple","mango","cherry"]
for i in list:
    print(i)
    
#use of break
list=["apple","mango","cherry"]
for i in list:
    #print(i)#apple,mango,cherry
    if(i=="mango"):
        break
    print(i)#apple
    
#use of continue
list=["apple","mango","cherry"]
for i in list:
    #print(i)#apple,mango,cherry
    if(i=="mango"):
        continue
    print(i)#apple,cherry
 
#range()
for i in range(2,6):
    print(i)
for i in range(2,30,3):
    print(i)
    
#nested loop->loop inside loop
list=["apple","mango","cherry"]
col=["red","yellow","redd"]
for x in col:
    for y in list:
        print(x,y)
        
#functions()->block of statements
def my_fun():
    print("Hello this is my function")
my_fun()
my_fun()   

#function with argument
def my_fun(name):
    print("Hello " +name)
my_fun("Rehan")
my_fun("Kamran")   
    
def my_fun(name,name1):
    print(name+" "+name1)
my_fun("Rehan","Attar")
     
#arbitatry arguments,*args
#if you don't know how many arguments that will be passes
#into your function
def my_fun(*kids):
    print(kids[0]+" "+kids[2])
my_fun("Hello","World","India")

#**kwargs argument->a keywors argument is where you provide a name
def my_fun(**kwargs):
    for key,value in kwargs.items():
        print("%s==%s"%(key,value))
my_fun(first="Rehan",mid="Samirkhan",last="Attar")

#default function
def my_fun(country="India"):
    print("I am from "+country+".")
my_fun("America")#if arrgument is passes it is used
my_fun()#other wise default 
my_fun("Singapore")

#we can pass list as argument
#any data type as an argument in a function
list=["apple","mango","cherry"]
def my_fun(list):
    for x in list:
        print(x)
my_fun(list)        

#function with return value
def my_fun(x):
    return x*5
my_fun(20)
print(f"Answer is {my_fun(20)} ")

#pass function->empty function
def my_fun():
    pass

#factorial of no
def fact(x):
    if x==1:
        return 1
    else:
        return(x*fact(x-1))
fact(3)

#lambda function->can have multiple arguments but only one expression
add=lambda a:a+10
print(add(20))

add=lambda a,b:a/b
print(add(20,5))

#finding odd no. in the list
lst=[23,5,22,11,21,12,4,6,8,67,3]
odd=list(filter(lambda x:(x%2 !=0),lst))
print(odd)
    
#square of no.     
lst=[5,20,4,6,8,67,3]
lst_sq=list(map(lambda x:(x**2),lst))
print(lst_sq)
  

##use case 1
'''Write a python code using logical operators and if elif.
so as to check height as well as
so as to allow for roller coaster also ask user age and charge
ticket accordingly'''

print("Welcome to the roller coaster")
height=int(input("Please enter your height in cm:"))
if height>=120:
    print("You are eligible for the Roller coaster")
    age=int(input("Please enter your age in years:"))
    bill=0
    if age<12:
        print("Childs ticket is 5 rs")
        bill=5
    elif 12>=age<=21:
        print("Ticket is 20 rs")
        bill=20
    elif age>21:
        print("Ticket is 50 rs")   
        bill=50
    else:
       print("You are under age")
    pc=input("Do you want popcorn Y or N:")
    if pc=="Y":
        bill+=3
        print(f"bill is {bill}")
    else:
        print(f"bill is {bill}")
    photo=input("Do you want photo Y or N")
    if photo=="Y":
        bill+=4
        print(f"Total bill is {bill}")
    else:
        print(f"Total bill is {bill}")
else:
    print("You are not eligible for the Roller coaster")
  

###use case 2
'''BMI calculator'''
height=float(input("Please enter your height in m:"))
weight=float(input("Please enter your weight in kg:"))
bmi=round((weight/(height*height)),2)
if bmi<18.5:
    print(f"You are under weight and your bmi is: {bmi}")
elif bmi>18.5 and bmi<25:
    print(f"You are normal weight and your bmi is: {bmi}")
elif bmi>25 and bmi<30:
    print(f"You are over weight and your bmi is: {bmi}")
elif bmi>30 and bmi<35:
    print(f"You are Obese and your bmi is: {bmi}")
elif bmi>35:
    print(f"You are clinically obese and your bmi is: {bmi}")


###use case 3
'''find duplicates in the list'''
list1=[1,2,3,20,4,5,20]  
list1.sort()
def dup(list1):
    for i in range(len(list1)-1):
        if (list1[i]==list1[i+1]):
            return True
    return False
print(dup(list1))

###leap year
def is_leapyear(year):
    if((year>0)and (year%4==0)and(year%100!=0)or(year%400==0)):
        return True
    return False
print(is_leapyear(2024))

###mario pyramid
#n=int(input("Enter no of lines"))
for i in range(4):
    for j in range(4):
        print("#",end=" ")
    print()
    
for i in range(4):
    for j in range(i+1):
        print("#",end=" ")
    print()    
    
    
for i in range(5):
     for j in range(5-i):
         print("#",end=" ")
     print()   
        
###diamondHW############################
for i in range(4):
    for j in range(i+2):
        print("#",end=" ")
    print()  
    
    
####################################### 


##minimum and maximum function
list=[1,3,2,5,4,7,6]
def min_max(list):
    min=list[0]
    for i in list:
        if i<min:
            min=i
    print("Minimum value is",min)   
    max=list[0]
    for i in list:
        if i>max:
            max=i
    print("Maximum value is",max)   
min_max(list)

###palindrome
def palindrome(n):
    if n==" ":
        print("Enter wrong string")
    else:
        str=n[::-1]
        if str==n:
            return True
    return False
palindrome(" ")      

#password
users=["Admin","Manager","Employee","staff"]
for user in users:    
    if user=="Admin":
        print("Hello admin")
    elif user=="Manager":
        print("Hello manager")
    elif user=="Employee":
        print("Hello Employee")
    else:
        print("Hello staff")
                 

#pass from user
user=input("Enter user:")   
if user=="Admin":
    print("Hello admin")
elif user=="Manager":
    print("Hello manager")
elif user=="Employee":
    print("Hello Employee")
elif user=="staff":
    print("Hello staff")
else:
    print("pass incorrect")    
    
##password picker
import string
adjs=["big","small","brave","pretty","heavy","proud",
      "red","dark"]    
nouns=["boy","laptop","goat","pen","monkey","bat",
       "leaf","bike","toy"]
import random
adj=random.choice(adjs)
noun=random.choice(nouns)  
number=random.randrange(0,100)
special=random.choice(string.punctuation)
password=adj+noun+str(number) +special
print("Your password is: %s"%password)          

###user choice password
import string
adjs=["big","small","brave","pretty","heavy","proud",
      "red","dark"]    
nouns=["boy","laptop","goat","pen","monkey","bat",
       "leaf","bike","toy"]
import random
print("Welcome to the password picker") 
while True:
    adj=random.choice(adjs)
    noun=random.choice(nouns)  
    number=random.randrange(0,100)
    special=random.choice(string.punctuation)
    password=adj+noun.upper()+str(number) +special
    print("Your password is: %s"%password) 
    response=input("Would yoy like to generate another passsword:")
    if response=="n":
        print("Your password is: %s"%password)
        break
    else:
        password=adj+noun.upper()+str(number) +special
        print("Your new password is: %s"%password) 
        




    
    
    



    