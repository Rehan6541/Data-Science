'''                          24/04/24                    '''
'''write a python program to draw a line with suitable 
label in the'''
import matplotlib.pyplot as plt
x=range(1,50)
y=[value*3 for value in x]
print("value of x:")
print(*range(1,50))
'''this is equivalence to i in range(1,50):
                     print(i,end='')'''
print("value of y(thrice of x):")
print(y)                     
#Plot lines and/or markers to the axes
plt.plot(x,y)
#set the x axis label of the current axis
plt.xlabel('x-axis')
#set the y axis label of the current axis
plt.ylabel('y-axis')
#Set a title
plt.title("Draw a line")
#display the figure
plt.show()



import matplotlib.pyplot as plt
#x axis values
x=[1,2,3]
#y axis values
y=[2,4,1]
#Plot lines and/or markers to the axes
plt.plot(x,y)
#set the x axis label of the current axis
plt.xlabel('x-axis')
#set the y axis label of the current axis
plt.ylabel('y-axis')
#Set a title
plt.title("sample graph")
#display the figure
plt.show()

'''write a python program to plot two or more lines on same
plot with suitable legends of each line'''
import matplotlib.pyplot as plt
##line 1 ponts
x1=[10,20,30]
y1=[20,40,10]

##line 2 ponts
x2=[10,20,30]
y2=[40,10,30]

#Plot the line 1 point
plt.plot(x1,y1,label="Line 1")
#Plot the line 2 point
plt.plot(x2,y2,label="Line 2")
#set the x axis label of the current axis
plt.xlabel('x-axis')
#set the y axis label of the current axis
plt.ylabel('y-axis')
#Set a title
plt.title("Two lines")
#Display the figure
plt.show()

'''write a python program to plot two or more lines on same
plot with suitable legends of each line with 
colurs,linewidth,linestyle'''
import matplotlib.pyplot as plt
##line 1 ponts
x1=[10,20,30]
y1=[20,40,10]

##line 2 ponts
x2=[10,20,30]
y2=[40,10,30]

#Plot the line 1 point
plt.plot(x1,y1,label="Line 1")
#Plot the line 2 point
plt.plot(x2,y2,label="Line 2")
#set the x axis label of the current axis
plt.xlabel('x-axis')
#set the y axis label of the current axis
plt.ylabel('y-axis')
#Set a title
plt.title("Two lines")
#display the figure
plt.plot(x1,y1,color='blue',linewidth=5,label="Line 1",linestyle='dashdot')
plt.plot(x2,y2,color='red',linewidth=5,label="Line 2",linestyle='dashed')
#show a legend on the plot
plt.legend()
#Display the figure
plt.show()

'''                          25/04/24                    '''
#introducing marker on graph
import matplotlib.pyplot as plt
#x axis values
x=[1,4,5,6,7]
#y axis values
y=[2,6,3,6,3]
#Plot lines and/or markers to the axes
plt.plot(x,y,color='red',linewidth=3,label="Line 1",
         linestyle='dashdot',marker='v',markerfacecolor='blue',markersize=12)
#set the x limits of the current axis
plt.xlim(1,8)
#set the y limits of the current axis
plt.ylim(1,8)
#naming the x axis
plt.xlabel('x-axis')
#naming the y axis
plt.ylabel('y-axis')
#giving a title to my graph
plt.title("line with marker")
#function to show
plt.show()

'''write a python program to display a bar chart of 
the popularity of programming language'''
import matplotlib.pyplot as plt
x=['java','python','php','javascript','c#','c++']
popularity=[22.2,17.6,8.8,8,7.7,6.7]
x_pos=[i for i,_ in enumerate(x)]
plt.bar(x_pos,popularity,color='red')
#naming the x axis
plt.xlabel('languages')
#naming the y axis
plt.ylabel('popularity')
#giving a title to my graph
plt.title("Popularity of progrm languages\n"+
          "worldwide,oct 2017 compared to a year ago")
plt.xticks(x_pos,x)
#turn on the grid
plt.minorticks_on()
plt.grid(which='major',linestyle='-',linewidth=0.5,color='blue')
#function to show
plt.show()

'''write a python program to display a bar chart
(horizontally) of the popularity of programming language'''
import matplotlib.pyplot as plt
x=['java','python','php','javascript','c#','c++']
popularity=[22.2,17.6,8.8,8,7.7,6.7]
x_pos=[i for i,_ in enumerate(x)]
plt.barh(x_pos,popularity,color='red')
#naming the x axis
plt.xlabel('languages')
#naming the y axis
plt.ylabel('popularity')
#giving a title to my graph
plt.title("Popularity of progrm languages\n"+
          "worldwide,oct 2017 compared to a year ago")
plt.yticks(x_pos,x)
#turn on the grid
plt.minorticks_on()
plt.grid(which='major',linestyle='-',linewidth=0.5,color='blue')
#function to show
plt.show()

'''write a python program to display a bar chart
of the popularity of programming language'''
#use uniform color
import matplotlib.pyplot as plt
x=['java','python','php','javascript','c#','c++']
popularity=[22.2,17.6,8.8,8,7.7,6.7]
x_pos=[i for i,_ in enumerate(x)]
plt.barh(x_pos,popularity,color=['red','black','blue','green','yellow','cyan'])
#naming the x axis
plt.xlabel('languages')
#naming the y axis
plt.ylabel('popularity')
#giving a title to my graph
plt.title("Popularity of progrm languages\n"+
          "worldwide,oct 2017 compared to a year ago")
plt.yticks(x_pos,x)
#turn on the grid
plt.minorticks_on()
plt.grid(which='major',linestyle='-',linewidth=0.5,color='blue')
#function to show
plt.show()


#Histogram
import matplotlib.pyplot as plt
blood_sugar=[113,85,90,150,149,88,93,115,80,77,82,120]
plt.hist(blood_sugar,rwidth=0.8)#by default number of bins set to 10
plt.hist(blood_sugar,rwidth=0.5,bins=4)

'''80-100=normal
100-125=pre diabetic
125 onwards=diabetic'''
plt.xlabel("sugar level")
plt.ylabel("number of patient")
plt.title("blood sugar chart")
plt.hist(blood_sugar,bins=[80,100,125,150],rwidth=0.8,color='green')

#Boxplot
import matplotlib.pyplot as plt
import numpy as np
#create a dataset
np.random.seed()
data=np.random.normal(100,20,200)
fig=plt.figure(figsize=(10,7))
#creating plot
plt.boxplot(data)
#show plot
plt.show()

#creating dataset
data1=np.random.normal(100,20,200)
data2=np.random.normal(90,20,200)
data3=np.random.normal(80,30,200)
data4=np.random.normal(70,40,200)
data=[data1,data2,data3,data4]
fig=plt.figure(figsize=(10,7))
#creating axis instance
ax=fig.add_axes([0,0,1,1])
#creating plot
bp=ax.boxplot(data)
#show plot
plt.show()






