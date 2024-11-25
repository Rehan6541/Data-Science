# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 08:58:20 2024

@author: Hp
"""
from bs4 import BeautifulSoup
soup=BeautifulSoup(open("C:\8-Text Mining\Text_MIning\sample_doc.html"),'html.parser')
print(soup)
#It is going to show all the html contents extracted
soup.text
#It will show only text
soup.content
#It is going to show all the contents extracted
soup.find('address')
soup.find_all('address')
soup.find_all('q')
soup.find_all('b')
table=soup.find('table')
table
for row in table.find_all('tr'):
    columns=row.find_all('td')
    print(columns)
    
#It will show all the rows expect first row
#Now we want to display M.tech which is located 
#in the third row
#I need to give [3][2]
    table.find_all('tr')[3].find_all('td')[2]


from bs4 import BeautifulSoup as bs
import requests
link="https://sanjivanicoe.org.in/index.php/contact"
page=requests.get(link)
page    
#<Response [200]> means that the connection is successfully established
page.content
#you will get all source code but very crowdy text
#let us apply html parser
soup=bs(page.content,'html.parser')
soup
#Now the text is clean but not upto the expectation
#Now let us apply prettify method
print(soup.prettify())
#The text is neat and clean
list(soup.children)
#finding all content using tab
soup.find_all('p')
#suppose you want to extract the content form first row


import bs4
from bs4 import BeautifulSoup as bs
import requests
link="https://www.amazon.in/product-reviews/B01EZ0X55C/ref=acr_dp_hist_5?ie=UTF8&filterByStar=five_star&reviewerType=all_reviews#reviews-filter-bar"
page=requests.get(link)
page
page.content
## now let us parse the html page
soup=bs(page.content,'html.parser')
print(soup.prettify())
#when you parse HTML using BeautifulSoup, you are converting the 
#raw HTML content of a web page into a structured format, 
#like a tree, where you can easily locate and manipulate individual 
#elements (such as tags, attributes, or text).

#page.content=> provides the raw HTML content,
#while soup.prettify()=> offers a formatted, human-readable version of the parsed HTML content.

## now let us scrap the contents
names=soup.find_all('span',class_='a-profile-name')
names
### but the data contains with html tags,let us extract names from html tags
cust_names=[]
for i in range(0,len(names)):
    cust_names.append(names[i].get_text())
    
cust_names
len(cust_names)
cust_name1=[]

for i in cust_names:
    if i not in cust_name1:
        cust_name1.append(i)
cust_name1
len(cust_name1)
cust_name1.pop(-1)
### There are total 11 users names 
#Now let us try to identify the titles of reviews
title=soup.find_all('a',class_='review-title')
title
# when you will extract the web page got to all reviews rather top revews.when you
# click arrow icon and the total reviews ,there you will find span has no class
# you will have to go to parent icon i.e.a
#now let us extract the data
review_titles=[]
for i in range(0,len(title)):
    review_titles.append(title[i].get_text())

review_titles
# ouput we will get consists of \n 
review_titles=[ title.strip('\n')for title in review_titles]
review_titles
len(review_titles)
##now let us scrap ratings
rating=soup.find_all('span',class_='a-icon-alt')
rating
###we got the data
rate=[]
for i in range(0,len(rating)):
    rate.append(rating[i].get_text())
rate
len(rate)   
rate1=[]

   
rate.pop(-1)
rate.pop(-1)
rate.pop(-1)
rate.pop(-1)
rate.pop(-1)
rate.pop(-1)
len(rate)
## now let us scrap review body
reviews=soup.find_all("span",class_='a-size-base review-text review-text-content')
reviews
review_body=[]
for i in range(0,len(reviews)):
    review_body.append(reviews[i].get_text())
review_body
review_body=[ reviews.strip('\n\n')for reviews in review_body]
review_body
len(review_body)
##########################################
###convert to csv file
import pandas as pd
df=pd.DataFrame()
df['customer_names']=cust_name1
df['review_title']=review_titles
df['rate']=rate
df['review_body']=review_body
df
df.to_csv('c:/360DG/Assignments/Text-minning/Amazon_reviews.csv',index=True)
########################################################
#sentiment analysis
import pandas as pd
from textblob import TextBlob
sent="This is very excellent garden"
pol=TextBlob(sent).sentiment.polarity
pol
df=pd.read_csv("c:/360DG/Assignments/Text-minning/Amazon_reviews.csv")
df.head()
df['polarity']=df['review_body'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)
df['polarity']    
    
    
    
    
    
    
    
    
    
    
    
    
    
    