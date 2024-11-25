# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 09:13:18 2024

@author: Hp
"""

import bs4
from bs4 import BeautifulSoup as bs
import requests
link="https://www.imdb.com/title/tt0068646/reviews"
page=requests.get(link)
page
page.content

## now let us parse the html page
soup=bs(page.content,'html.parser')
print(soup.prettify())

## now let us scrap the contents
#Now let us try to identify the titles of reviews
title=soup.find_all('a',class_='title')
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
review_titles[:]=[ title.strip('\n')for title in review_titles]
review_titles
len(review_titles)
#Got 25 review titles


##now let us scrap ratings
rating=soup.find_all('span',class_='point-scale')
rating
###we got the data
rate=[]
for i in range(0,len(rating)):
    rate.append(rating[i].get_text())
rate
rate[:]=[ r.strip('/')for r in rate]
rate
len(rate)
rate.append('')
rate.append('')
len(rate)
#Got 25 ratings



##now let us review body
review=soup.find_all('div',class_='text show-more__control')
review
###we got the data
review_body=[]
for i in range(0,len(review)):
    review_body.append(review[i].get_text())
review_body
review_body=[ reviews.strip('\n\n')for reviews in review_body]
len(review_body)

###convert to csv file
import pandas as pd
df=pd.DataFrame()
df['review_title']=review_titles
df['rate']=rate
df['review_body']=review_body
df
df.to_csv("C:\8-Text Mining\Text_MIning\Amazon_reviews.csv",index=True)
########################################################

   
#sentiment analysis
import pandas as pd
from textblob import TextBlob
sent="This is very excellent garden"
pol=TextBlob(sent).sentiment.polarity
pol
df=pd.read_csv("C:\8-Text Mining\Text_MIning\Amazon_reviews.csv")
df.head()
df['polarity']=df['review_body'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)
df['polarity']













