# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 08:43:35 2024

@author: Hp
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder

#Sample dataset
transactions=[
    ['Milk','Bread','Butter'],
    ['Bread','Eggs'],
    ['Milk','Bread','Eggs','Butter'],
    ['Bread','Eggs','Butter'],
    ['Milk','Bread','Eggs']
    ]
#Step1:Convert the dataset into a format for Apriori
te=TransactionEncoder()
te_ary=te.fit(transactions).transform(transactions)
df=pd.DataFrame(te_ary,columns=te.columns_)

#Step2:Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets=apriori(df,min_support=0.5,use_colnames=True)

#Step3:Generate association rules from the frequent itemsets
rules=association_rules(frequent_itemsets,metric="lift",min_threshold=1)

#Step4:Output the results
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules[['antecedents','consequents','support','confidence','lift']])


import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder

#Step1:Simulating healthcare trancations (symptoms/disease/treatements)
healthcare_data=[
    ['fever','cough','covid-19'],
    ['cough','sore throat','flu'],
    ['fever','cough','shortness of breath','covid-19'],
    ['cough','sore throat','flu','headache'],
    ['fever','body ache','flu'],
    ['fever','cough','covid-19','shortness of breath'],
    ['sore throat','headache','cough'],
    ['body ache','fatigue','flu']
    ]

#Step1:Convert the dataset into a format for Apriori
te=TransactionEncoder()
te_ary=te.fit(healthcare_data).transform(healthcare_data)
df1=pd.DataFrame(te_ary,columns=te.columns_)

#Step2:Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets=apriori(df1,min_support=0.3,use_colnames=True)

#Step3:Generate association rules from the frequent itemsets
rules=association_rules(frequent_itemsets,metric="confidence",min_threshold=0.7)

#Step4:Output the results
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules[['antecedents','consequents','support','confidence','lift']])



















