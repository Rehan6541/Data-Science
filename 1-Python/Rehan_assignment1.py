##########1 mailing address


'''Write a program that displays your name and complete
mailing address formatted in the manner that you
would usually see it on the outside of an envelope.
your program does not need to read any input from user'''

x='Rehan Attar'
print(x)
y='128,Niwara road'
print(y)
z='Kopargaon ,423601'
print(z)

############2 Area of room


'''Write a program that asks the user to enter the width and length of a room. Once
the values have been read, your program should compute and display the area of the
room. The length and the width will be entered as floating point numbers. Include
units in your prompt and output message; either feet or meters, depending on which
unit you are more comfortable working with.'''

w=float(input("Enter the width of the room(in meters):"))
l=float(input("Enter the length of the room(in meters):"))
a=l*w
print("The area of square is",a,"square meters")

###########3 Area of field


'''Create a program that reads the length and width of a farmer’s field from the user in
feet. Display the area of the field in acres.
Hint: There are 43,560 square feet in an acre.'''

w=int(input("EEnter the length of farmers field (in feet):"))
area=l*w
acres=area/43560
print("Area of farners field",acres,"  acre")

##########4 Bottle Deposits


'''In many jurisdictions a small deposit is added to drink containers to encourage people
to recycle them. In one particular jurisdiction, drink containers holding one liter or
less have a $0.10 deposit, and drink containers holding more than one liter have a
$0.25 deposit.
Write a program that reads the number of containers of each size from the user.
Your program should continue by computing and displaying the refund that will be
received for returning those containers. Format the output so that it includes a dollar
sign and always displays exactly two decimal places'''

a=int(input("Enter the number of conatiners holding the capacity of one litre or less than one liter:"))
b=int(input("Enter the number of conatiners holding the capacity of more than one litre:"))
refund = round((a * 0.10) + (b * 0.25),2)
print(f"The refund you will get is ${refund}")

#########5Tax and Tip


'''The program that you create for this exercise will begin by reading the cost of a meal
ordered at a restaurant from the user. Then your program will compute the tax and
tip for the meal. Use your local tax rate when computing the amount of tax owing.
Compute the tip as 18 percent of the meal amount (without the tax). The output from
your program should include the tax amount, the tip amount, and the grand total for
the meal including both the tax and the tip. Format the output so that all of the values
are displayed using two decimal places.'''

meal=float(input("Enter the cost of meal:"))
tax_rate = 0.07
tax=tax_rate*meal
tip_rate = 0.18
tip=tip_rate*meal
total_cost= meal + tax + tip_rate
print(f"Tax amount you will get is ${tax}")
print(f"Tip amount you will get is ${tip}")
print(f"Total cost you will get is ${total_cost}")

#6Height Units


'''Many people think about their height in feet and inches, even in some countries that
primarily use the metric system. Write a program that reads a number of feet from
the user, followed by a number of inches. Once these values are read, your program
should compute and display the equivalent number of centimeters'''




feet=float(input("Enter the number of feet:"))
inches=float(input("Enter the number of inches:"))
feet_cm = feet*30.48
inches_cm = inches*2.54
total_cm = feet_cm + inches_cm
print(f" Total height in centimeter is {total_cm}")











