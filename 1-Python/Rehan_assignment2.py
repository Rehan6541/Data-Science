#Q.1
'''Write a program that reads a letter of the alphabet from the
user. If the user enters a, e, i, o or u then your program should display a message
indicating that the entered letter is a vowel. If the user enters y then your program
should display a message indicating that sometimes y is a vowel, and sometimes y is
a consonant. Otherwise your program should display a message indicating that the
letter is a consonant.'''

alpha=input("Enter any alphabet:")
vowels=["a","e","i","o","u"]
if alpha.lower() in vowels:
    print("The entered letter is a vowel.")
elif alpha.lower() == 'y':
    print("Sometimes 'y' is a vowel, and sometimes 'y' is a consonant.")
else:
    print("The entered letter is a consonant.")
    
    
#Q.2
'''Write a program that determines the name of a shape from its number of sides. Read
the number of sides from the user and then report the appropriate name as part of
a meaningful message. Your program should support shapes with anywhere from 3
up to (and including) 10 sides. If a number of sides outside of this range is entered
then your program should display an appropriate error message.'''
 
side=int(input("Enter no of side:"))
print(f"Side entered is {side}.")
def get_shape(side):
    shapes ={3: "triangle",
            4: "quadrilateral",
            5: "pentagon",
            6: "hexagon",
            7: "heptagon",
            8: "octagon",
            9: "nonagon",
            10: "decagon"}
    if side<2 or side>10:
        print("Error")
    else:
        print(shapes[side])
get_shape(side) 


#Q.3
'''The length of a month varies from 28 to 31 days. In this exercise you will create
a program that reads the name of a month from the user as a string. Then your
program should display the number of days in that month. Display “28 or 29 days”
for February so that leap years are addressed.'''


mon=(input("Enter name of month:"))
print(f"Month entered is {mon}.")
def get_day(mon):
    month_days = {"january":31,
                  "february":"28 or 29",
                  "march":31,
                  "april":30,
                  "may":31,
                  "june":30,
                  "july":31,
                  "august":31,
                  "september":30,
                  "october":31,
                  "november":30,
                  "december":31}
    if mon.lower() in month_days:
        print(month_days[mon])
    else:
        print("Error")
get_day(mon)        
        
#Q.4
'''. A triangle can be classified based on the lengths of its sides as equilateral, isosceles
or scalene. All 3 sides of an equilateral triangle have the same length. An isosceles
triangle has two sides that are the same length, and a third side that is a different
length. If all of the sides have different lengths then the triangle is scalene.
Write a program that reads the lengths of 3 sides of a triangle from the user.
Display a message indicating the type of the triangle.'''
side1=int(input("Enter the 1st side:"))
side2=int(input("Enter the 2nd side:"))
side3=int(input("Enter the 3rd side:"))
if(side1==side2==side3):
    print("Equilateral Triangle")
elif(side1==side2 or side2==side3 or side1==side3):
    print("Isosceles Triangle")
else:
    print("Scalene Triangle")
    
#Q.5
'''The year is divided into three seasons: summer, rainy and winter. While the
exact dates that the seasons change vary a little bit from year to year because of the
way that the calendar is constructed, Write a program to display the season if date is given.'''

month=int(input("enter month from 1 to 12:"))
date=int(input("enter date from 1 to 31:"))
if((month==2 and date>=25) or month==3 or month==4 or month==5 or (month==6 and date<=5)):
    print("its a summer season")
elif((month==6 and date>1) or month==7 or month==8 or (month==9 and date>5)):
    print("its rainy season")
else:
    print("its winter season")
    





   