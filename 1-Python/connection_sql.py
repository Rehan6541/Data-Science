#Thu Jul 11 10:39:17 2024

#After insatlling with pip install pyscopg2
import psycopg2 as pg2

#Create a connection with postgresql
#'password' is whatever password you set,
conn=pg2.connect(database='dvdrental',user='postgres',password='@Rehan1234')

#Establish connection and start cursor to be ready to query
cur=conn.cursor()

#Pass in a postgresql query as a string
cur.execute("SELECT *FROM payment")

#Return a tuple of the first row as python object
cur.fetchone()

#Return N number of rows
cur.fetchmany(10)

#Return all rows at once
cur.fetchall()

#to save and index result,assign it to a variable
data=cur.fetchmany(10)

conn.close