##############ADVANCE PYTHON################
        
####list comprehension####
list=[]
for num in range(0,20):
    list.append(num)
print(list)

##by list comprehension
list=[num for num in range(0,20)]
print(list)


names=["dada","mama","kaka"]
list=[name.capitalize() for name in names]
print(list)

#comprehension with if statement
def is_even(num):
    return num%2==0
list=[num for num in range(21) if is_even(num)]
print(list)

#comphrension with nested for loop 
list=[f"{x}{y}"for x in range(3) for y in range(3)]
print(list)

##dict comprehension
dict={x:x*x for x in range(3)}
print(dict)

###Generator
'''It is another way of creating iterators in a simple way
where it uses the key word "yield".Instead of returning
it in a defined function.
Generators are implemented using a function'''

#generator for all values
gen=(x for x in range(3))
print(gen)
for num in gen:
    print(num)

#generator for single values
gen=(x for x in range(3))
next(gen)
next(gen)

##functions which return multiple values
def range_even(end):
    for num in range(0,end,2):
        yield num
for num in range_even(6):
    print(num)
    
#now instead of usimg for loop we can write create generators
gen=range_even(6)
next(gen)
next(gen)
    
###chaining generators
def lengths(itr):
    for ele in itr:
        yield len(ele)
def hide(itr):
    for ele in itr:
        yield ele*"*"
        
password=["not good","give m-pass"]    
for password in hide(lengths(password)):
    print(password)
    
        
'''"ele*" appears to be a placeholder for an element
from an iterable.The asterisk(*) is likely just a character
used to represent a placeholder or a wildcard.
for instance if u r iteratngover a lst of elemts,"ele*"
could symbolize any representation that does not  correspond to any spevific syntax
in python or itertolls'''


##take password from user and hide it
adj=input("Enter an adj:")
noun=input("Enter a noun:")
number=input("Enter a number:")
sc=input("Enter a special character:")
password=adj+noun+str(number) +sc
print("Your password is: %s"%password) 
def lengths(itr):
    for ele in itr:
        yield len(ele)
def hide(itr):
    for ele in itr:
        yield ele* "*"
password=adj+noun+str(number)+sc    
for password in hide(lengths(password)):
    print(password,end="") 
    
#########################25/03/24    
####Enumerator
##printing list with index
list=["Apple","Mango","Wmelon"]
for index in range(len(list)):
    print(f"{index+1}-{list[index]}")
 
#same code can be implemented using enumerate
list=["Apple","Mango","Wmelon"]
for index,item in enumerate(list,start=6):
    print(f"{index}-{item}")    
    
#use of zip function    
list1=["Dada","Mama","Kaka"]
mob=["9975","8648","9327"]
for nm,li  in zip(mob,list1):
        print(f"{nm}-{li}")    
    
#use of zip function with mis match list
from itertools import zip_longest
list1=["Dada","Mama","Kaka","Baba"]
mob=["9975","8648","9327"]
for nm,li  in zip_longest(mob,list1):
    print(f"{nm}-{li}") 
    
##use of fill value instead none
from itertools import zip_longest
list1=["Dada","Mama","Kaka","Baba"]
mob=["9975","8648","9327"]
for nm,li  in zip_longest(mob,list1,fillvalue=0):
    print(f"{nm}-{li}") 

#use of all(),if all the values are true then it will
#produce output  
list=[2,3,4,1,0,-2,7,-4]#values must be non zero,+ve,-ve
if all(list):
    print("all values are true")
else:
    print("there are non zero values")
    
list=[2,3,4,1,0,-2,7,-4]#values must be non zero,+ve,-ve
if all(list):
    print("all values are true")
else:
    print("there are non zero values")
        
    
#use of any(),if any one non zero values
list=[2,3,4,1,0,-2,0,7,-4]
if any(list):
    print("true")
else:
    print("useless")
        

list=[0,0,0,0,0]
if any(list):
    print("it has some non zero values")
else:
    print("useless,all values are null")    
    
##count()
from itertools import count
counter=count()
print(next(counter))#starts from 0
print(next(counter))
print(next(counter))

#now let us start from 1
from itertools import count
counter=count(start=1)
print(next(counter))#starts from 1
print(next(counter))
print(next(counter))

#cycle()
#suppose you have repeated tasks to be done,then
import itertools
instruct=("eat","code","sleep","repeat")
for instr in itertools.cycle(instruct):
    print(instr)

##repeat()
from itertools import repeat
for msg in repeat("keep patience",times=4):
    print(msg)
    
###combination()
from itertools import combinations
players=["Virat","Dhoni","Rohit"]
for i in combinations(players,2):
    print(i)
    
#permutations()
from itertools import permutations
players=["Virat","Dhoni","Rohit"]
for i in permutations(players,2):
      print(i) 
    
#product()
from itertools import product
team_a=["Virat","Dhoni","Rohit"]
team_b=["Shami","Bumrah","Ashwin"]
for pair in product(team_a,team_b):
    print(pair)
    

###shallow and deep copy
'''In python assignment statement (a=b) do not create
real copies
it only creates a new variable with same reference.
so when u want to make actual copies of mutable objects
(list,dicts)
and want to modify the copy without affecting the original
you have to be careful.
for "real"copies we can use the copies module.
however,for compound/nested objects (eg nested lists or dicts) and
custom objects there is an imp difference between shallow
and deep copying.

shallow copy->only one level deep.it creates a new 
copy and populates it with references to the nested objects.
this means modyfying a nested obj in the copy deeper than modification in original also
deep copy->a full independent clone.it creates a new 
copy and then recusively populates it with copies of 
nested'''

#assignment operation
#This will create a new variable with the same reference
list_a=[1,2,3,4,2]
list_b=list_a
list_a[0]=-10
print(list_a) 
print(list_b)    


##Shallow copyyy
import copy
list_a=[1,2,3,4,2]#one level
list_b=copy.copy(list_a)
#not affect the other list
list_b[0]=-10
print(list_a) 
print(list_b)

#with nested object modifying level two
list_a=[[1,2,3,4,2],[2,3,4,5]]#two level
list_b=copy.copy(list_a)
list_a[0][0]=-10
print(list_a) 
print(list_b)
    
##Deep copy
list_a=[[1,2,3,4,2],[2,3,4,5]]#two level
list_b=copy.deepcopy(list_a)
list_a[0][0]=-10
print(list_a) 
print(list_b)
