#Created on Wed Jun 12 2024
##PYTHON_NLP

sent="we are learning textmining from sanjivani AI"
#it will show position of learning
sent.index("learning")
#it will the words and show position of from
sent.split().index("from")
#it will reverse the word learning
#[start:end end:-1(start)]
sent.split()[2][::-1]

#if you want to print first and last word
words=sent.split()#tokenisation
first_word=words[0]
first_word
last_word=words[-1]
last_word

#if you want to concate first and last word
concate_word=first_word+" "+last_word
concate_word

#if we want to print even words from the sentence
[words[i] for i in range(len(words)) if i%2==0]

#display only AI
sent[-3:]

#full sentence in reverse by characterwise and wordwise also
sent[::-1]

#full sentence in reverse (each word in reverse)
print(" ".join(word[::-1]for word in words))

#tokenization
import nltk
nltk.download('punkt')
from nltk import word_tokenize
words=word_tokenize("I am reading NLP fundamentals")
print(words)

#parts of speech (Pos) tagging
nltk.download('averaged_perceptron_tagger')
nltk.pos_tag(words)
#it will mention parts of speech

#stop words from nltk library
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words=stopwords.words('English')#179 stopwords will display
stop_words

#####Thrus Jun 13 2024

#replace certain word from sentence
sent2="I visited MY from IND MY on 14-02-19"
normalised_sent=sent2.replace("MY","Malayasia").replace("IND","India")
normalised_sent=normalised_sent.replace("-19","-20")
normalised_sent

#autocorrect the word
from autocorrect import Speller
#declare the function Speller defined for english
spell=Speller(lang='en')
spell("Engilisgh")
spell("campues")

#autocorrect the sentence
import nltk
nltk.download('punkt')
from nltk import word_tokenize
sent3="Ntural Lanagage processin deals withh the aart of extracting"
#tokenize the sent3
sent3=word_tokenize(sent3)
correct_sent=" ".join([spell(word)for word in sent3])
correct_sent

#stemming
stemmer=nltk.stem.PorterStemmer()
stemmer.stem("programming")
stemmer.stem("jumped")
stemmer.stem("coordinating")

#lematizer
#lematizer looks into dictionary words
import nltk
nltk.download("wordnet")
from nltk.stem.wordnet import WordNetLemmatizer
lematizer=WordNetLemmatizer()
lematizer.lemmatize("programed")
lematizer.lemmatize("coordinating")


#chunking->identifies named entities
nltk.download("maxent_ne_chunker")
nltk.download("words")
sent4="We are learning NLP in python by SanjivaniAI"
#tokenize sent4
nltk.download('averaged_perceptron_tagger')
words=word_tokenize(sent4)
words=nltk.pos_tag(words)
i=nltk.ne_chunk(words,binary=True)
[a for a in i if len(a)==1]

#Sentence tokenization
from nltk import sent_tokenize
sentt=sent_tokenize("We are learning NLP in python. Do you knoe where it is located? Yes u are right. It is in Kopargaon")
sentt

#lesk->if there are to similar words this gives to two diff meaning
from nltk.wsd import lesk
sentt1="keep your savings in the bank"
print(lesk(word_tokenize(sentt1),"bank"))
sentt2="It is risky to drive over the banks of river"
print(lesk(word_tokenize(sentt2),"bank"))

##Friday 14-06-24
""" removing special character
special character as you know are non-aplhanumeric character.
these character are most often found in comments references currency number etc
these character add no values to text understanding and induce noise ito alogorithm
for that regex pacakage is used"""

import re# enivornment > pip install regex
chat1='Hello I am having an issue with my order #412889912'
pattern='order[^\d]*(\d*)'
matches=re.findall(pattern,chat1)
matches

chat2='I have a problem with my order number 412889912'
pattern ='order[^\d]*(\d*)'
matches=re.findall(pattern,chat2)
matches

chat3='my order 412889912 is having an issue, i was charged 300$ but online it says 280$'
pattern='order[^\d]*(\d*)'
matches=re.findall(pattern,chat2)
matches

# if we dont want to use re.findall 
def get_pattern_match(pattern,text):
    matches=re.findall(pattern,text)
    if matches:
        return matches[0]
get_pattern_match('order[^\d]*(\d*)',chat1)

#18-06-2004
#how to extract email from chat
chat1='you ask lot of questions 12345678912,abc@xyz.com'
chat2='here it is:(123)-567-8912,abc@xyz.com'
chat3='yes,phone:12345678912 email:abc@xyz.com'
get_pattern_match('[a-zA-Z0-9_]*@[a-z]*\.[a-zA-Z0-9]*',chat1)
get_pattern_match('[a-zA-Z0-9_]*@[a-z]*\.[a-zA-Z0-9]*',chat2)
get_pattern_match('[a-zA-Z0-9_]*@[a-z]*\.[a-zA-Z0-9]*',chat3)
#output:Out[21]: 'abc@xyz.com'
#how to extract phone no from chat
get_pattern_match('(\d{10})|(\(\d{3}\)-\d{3}-\d{4})',chat1)
get_pattern_match('(\d{10})|(\(\d{3}\)-\d{3}-\d{4})',chat2)
get_pattern_match('(\d{10})|(\(\d{3}\)-\d{3}-\d{4})',chat3)
#output:1.('1234567891', '')
#putput:2('', '(123)-567-8912')
#output:3('1234567891', '')

text="""Born	Elon Reeve Musk
June 28, 1971 (age 52)
Pretoria, Transvaal, South Africa
Citizenship	
South Africa
Canada
United States
Education	University of Pennsylvania (BA, BS)
Title	
Founder, CEO, and chief engineer of SpaceX
CEO and product architect of Tesla, Inc.
Owner, CTO and Executive Chairman of X (formerly Twitter)
President of the Musk Foundation
Founder of The Boring Company, X Corp., and xAI
Co-founder of Neuralink, OpenAI, Zip2, and X.com (part of PayPal)
Spouses	
Justine Wilson
​
​(m. 2000; div. 2008)​
Talulah Riley
​
​(m. 2010; div. 2012)​
​
​(m. 2013; div. 2016)​"""

get_pattern_match(r'age (\d+)', text)
get_pattern_match(r'Born(.*)\n',text).strip()
get_pattern_match(r'Born.*\n(.*)\(age',text).strip()
get_pattern_match(r'\(age.*\n(.*)',text)


###Wednesday 19-06-24
def get_personal_info(text):
    age=get_pattern_match(r'age (\d+)', text)
    full_name=get_pattern_match('Born(.*)\n', text)
    birth_date=get_pattern_match('Born.*\n(.*)\(age',text)
    birth_place=get_pattern_match('\(age.*\n(.*)', text)
    return {
        'age': int(age),
        'name': full_name.strip(),
        'DOB': birth_date.strip(),
        'Birth Place': birth_place.strip()
        }
text='''Born	Elon Reeve Musk
June 28, 1971 (age 52)
Pretoria, Transvaal, South Africa
Citizenship	
South Africa
Canada
United States
Education	University of Pennsylvania (BA, BS)
Title	
Founder, CEO, and chief engineer of SpaceX
CEO and product architect of Tesla, Inc.
Owner, CTO and Executive Chairman of X (formerly Twitter)
President of the Musk Foundation
Founder of The Boring Company, X Corp., and xAI
Co-founder of Neuralink, OpenAI, Zip2, and X.com (part of PayPal)
Spouses	
Justine Wilson
​
​(m. 2000; div. 2008)​
Talulah Riley
​
​(m. 2010; div. 2012)​
​
​(m. 2013; div. 2016)​
Partners	
Grimes (2018–2021)[1]
Children	11[a][3]
Parents	
Errol Musk
Maye Musk
Relatives	
Kimbal Musk (brother)
Tosca Musk (sister)
Lyndon Rive (cousin)
Signature'''
get_personal_info(text)

def get_personal_info(text):
    age=get_pattern_match(r'age (\d+)', text)
    full_name=get_pattern_match('Born(.*)\n', text)
    birth_date=get_pattern_match('Born.*\n(.*)\(age',text)
    birth_place=get_pattern_match('\(age.*\n(.*)', text)
    return {
        'age': int(age),
        'name': full_name.strip(),
        'DOB': birth_date.strip(),
        'Birth Place': birth_place.strip()
        }
text='''Ambani on an Indian postal stamp issued in 2002
Born	Dhirubhai Ambani
28 December 1932(age 92)
Chorwad, Junagadh State, British India
(present-day Gujarat, India)
Died	6 July 2002 (aged 69)
Mumbai, Maharashtra, India
Citizenship	British India (1932–1947)
Dominion of India (1947–1950)
India (1950–2002)
Occupation	Businessman
Organization(s)	Reliance Industries
Reliance Capital
Reliance Infrastructure
Reliance Power
Spouse	Kokila Dhirubhai Ambani
​
​(m. 1955)​
Children	4, including Mukesh Ambani and Anil Ambani
Awards	Padma Vibhushan (2016) (posthumously)'''
get_personal_info(text)

##info from pdf
from PyPDF2 import PdfFileReader
#importing required Pdfreader
from PyPDF2 import PdfReader
#creating a pd
reader=PdfReader("kopargaon-part-1.pdf")
print(reader)

#printing no of pages
print(len(reader.pages))

#Getting a specific page from the pdf file
page=reader.pages[1]

#Extracting text from pages
text=page.extract_text()
print(text)

#matrix
reader=PdfReader("matrix_basics.pdf")
print(reader)

#printing no of pages
print(len(reader.pages))

#Getting a specific page from the pdf file
page=reader.pages[1]

#Extracting text from pages 
text=page.extract_text()
print(text)

##Thrusday 20/06/24
#Extracting words from sentence
import re
sentence="sharad twitted , wittnessing 68th republic day India from Rajpath , \new Delhi,Mesmorizing performance by Indian Army!"
re.sub(r'([^\s\w]/_)+','',sentence).split()

#Extracting n-grams from sentence
import re
def n_grams_extracter(input_str,n):
    tokens=re.sub(r'([^\s\w]/_)+','',sentence).split()
    for i in range(len(tokens)-n+1):
        print(tokens[i:i+n])
n_grams_extracter("The cute little boy is playing with pussy",2)       
n_grams_extracter("The cute little boy is playing with pussy",3)

#Extracting all twitter handles from following text
text='''Follow our leader Elon musk on twitter here: https://twitter.com/elonmusk, more information on Tesla's products can be found at https://www.tesla.com/. Also here are leading influencers for tesla related news,https://twitter.com/teslarati, https://twitter.com/dummy_tesla https://twitter.com/dummy_2_tesla'''
pattern='https://twitter.com/([\w_]+)'
re.findall(pattern,text)

#Extracting particular word
text='''
Concentration of Risk: Credit Risk
Financial instruments that potentially subject us to a concentration of credit risk consist of cash, cash equivalents, marketable securities,
restricted cash, accounts receivable, convertible note hedges, and interest rate swaps. Our cash balances are primarily invested in money market funds
or on deposit at high credit quality financial institutions in the U.S. These deposits are typically in excess of insured limits. As of September 30, 2021
and December 31, 2020, no entity represented 10% or more of our total accounts receivable balance. The risk of concentration for our convertible note
hedges and interest rate swaps is mitigated by transacting with several highly-rated multinational banks.
Concentration of Risk: Supply Risk
We are dependent on our suppliers, including single source suppliers, and the inability of these suppliers to deliver necessary components of our
products in a timely manner at prices, quality levels and volumes acceptable to us, or our inability to efficiently manage these components from these
suppliers, could have a material adverse effect on our business, prospects, financial condition and operating results.'''
pattern='Concentration of Risk:([^\n]*)'
re.findall(pattern,text)

#Extract number orders
text = '''
Tesla's gross cost of operating lease vehicles in FY2021 Q1 S2 was $4.85 billion.
BMW's gross cost of operating vehicles in FY2021 S1 was $8 billion.
'''
pattern='FY(\d{4} (?:Q[1-4]|S[1-2]))'
#?:->match this and | a or b
matches=re.findall(pattern,text)
matches

#Extracting mobile number
text='''
Elon musk's phone number is 9991116666, call him if you have any questions on dodgecoin. Tesla's revenue is 40 billion
Tesla's CFO number (999)-333-7777
'''
pattern='\(\d{3}\)-\d{3}\-\d{4}|\d{10}'
matches=re.findall(pattern,text)
matches

#Extracting notes names from this text
text = '''
Note 1 - Overview
Tesla, Inc. (“Tesla”, the “Company”, “we”, “us” or “our”) was incorporated in the State of Delaware on July 1, 2003. We design, develop, manufacture and sell high-performance fully electric vehicles and design, manufacture, install and sell solar energy generation and energy storage
products. Our Chief Executive Officer, as the chief operating decision maker (“CODM”), organizes our company, manages resource allocations and measures performance among two operating and reportable segments: (i) automotive and (ii) energy generation and storage.
Beginning in the first quarter of 2021, there has been a trend in many parts of the world of increasing availability and administration of vaccines
against COVID-19, as well as an easing of restrictions on social, business, travel and government activities and functions. On the other hand, infection
rates and regulations continue to fluctuate in various regions and there are ongoing global impacts resulting from the pandemic, including challenges
and increases in costs for logistics and supply chains, such as increased port congestion, intermittent supplier delays and a shortfall of semiconductor
supply. We have also previously been affected by temporary manufacturing closures, employment and compensation adjustments and impediments to
administrative activities supporting our product deliveries and deployments.
Note 2 - Summary of Significant Accounting Policies
Unaudited Interim Financial Statements
The consolidated balance sheet as of September 30, 2021, the consolidated statements of operations, the consolidated statements of
comprehensive income, the consolidated statements of redeemable noncontrolling interests and equity for the three and nine months ended September
30, 2021 and 2020 and the consolidated statements of cash flows for the nine months ended September 30, 2021 and 2020, as well as other information
disclosed in the accompanying notes, are unaudited. The consolidated balance sheet as of December 31, 2020 was derived from the audited
consolidated financial statements as of that date. The interim consolidated financial statements and the accompanying notes should be read in
conjunction with the annual consolidated financial statements and the accompanying notes contained in our Annual Report on Form 10-K for the year
ended December 31, 2020.
'''
pattern='Note \d - ([^\n]*)'
matches=re.findall(pattern,text)
matches

#Extract Financial period
text = '''
The gross cost of operating lease vehicles in FY2021 Q1 was $4.85 billion.
In previous quarter i.e. FY2020 Q4 it was $3 billion. 
'''
pattern='FY(\d{4} Q[1-4])'
matches=re.findall(pattern,text)
matches

text = '''
The gross cost of operating lease vehicles in FY2021 Q1 was $4.85 billion.
In previous quarter i.e. fy2020 Q4 it was $3 billion. 
'''
pattern='FY(\d{4} Q[1-4])'
matches=re.findall(pattern,text,flags=re.IGNORECASE)
matches


##Assignment Review
text='''
The market price of the book is Rs.200 but I bought it for Rs.150.
war is peace.
Freedom is slavery.
Ignorance is strength.
A beautiful novel by author George Orwell heloo hii.
We humans are gifted with most powerful mind but our human society develops on how we use it.
As human is gifted with emotions like love and care we humans also have emotions like greed and hatred.
This novels tells you all about it
It is story of of place known as Oceania which is under dictatorship in which Winston smith is a person who works in Government ministry but always hates the government but all other praise or show that they are prasing the government
He then falls in love with jis co worker Julia.
but their love affairs comes to end when get caught and then series of tourcher processes occur
After reading this book i felt that we are very luck that we live in democracy by worker tom.'''

#Tokenize
from nltk import sent_tokenize
sentt=sent_tokenize(text)
sentt
text.split('\n')

#text extraction
import re
def get_pattern_match(pattern,text):
    matches=re.findall(pattern,text)
    if matches:
        return matches
get_pattern_match('Rs.(\d+)', text)
get_pattern_match('author ([^\n]*)', text)

#Word replace
normalised_sent=text.replace("Julia","Jesica").replace("tom","Tommy")
normalised_sent

#Assignment
text='''Thanks Flipkart for delivering the product in ontime. 
The color of the watch is not pink... On other platforms like amazon
the same watch was at Rs.1299 i bought it from flipkart just at
Rs.1000.
it is available in .cream color,black color,white color.
This is a cream color.... But in day and nght this are
 different color shade are shown... Nice watch .
 Also good deisgned include good fetures. 
 The all watch faces including ediing ysstem are done in noise fit App. 
 '''
import nltk 
from nltk import word_tokenize
word=word_tokenize(text)
word

from nltk import sent_tokenize
sent=sent_tokenize(text)
sent

import re
re.sub(r'([^\s\w]_)+', ' ', text).split()

from autocorrect import Speller
spell=Speller(lang='en')
spell(text)

rep=text.replace("noise","Boat")
rep

rep=re.sub(r'Noise','Boat',text,flags=re.IGNORECASE)
rep

pattern='Rs.(\d+)'
matches=re.findall(pattern,text)
matches

pattern='([^\n]*) color'
matches=re.findall(pattern,text)
matches


##Friday 21/06/24
from nltk import ngrams
#extraction n-grams with nltk
list(ngrams("The cute little boy is playing with kitten".split(),2))
list(ngrams("The cute little boy is playing with kitten".split(),4))

from textblob import TextBlob#anagramy
blob=TextBlob("The cute little boy is playing with kitten")
blob.ngrams(n=2)
blob.ngrams(n=3)

#Tokenization using TextBlob
from textblob import TextBlob
sentence="sharad twitted , wittnessing 68th republic day India from Rajpath , \new Delhi,Mesmorizing performance by Indian Army!"
blob=TextBlob(sentence)
blob.words

#Tweet Tokenizer
from nltk.tokenize import TweetTokenizer
tweet_token=TweetTokenizer()
tweet_token.tokenize(sentence)

##Multi-Word_Expression
from nltk.tokenize import MWETokenizer
mwe_token=MWETokenizer([('republic','day')])
mwe_token.tokenize(sentence.split())
mwe_token.tokenize(sentence.replace('!',' ').split())

#Tokenization using Keras
sentence="sharad twitted , wittnessing 68th republic day India from Rajpath , \new Delhi,Mesmorizing performance by Indian Army!"
from tensorflow.keras.preprocessing.text import text_to_word_sequence
text_to_word_sequence(sentence)











