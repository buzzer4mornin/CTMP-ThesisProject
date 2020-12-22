import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df=pd.read_csv('SMSSpamCollection', sep='\t', names=['Status','Message'])

df.loc[df["Status"]=='ham',"Status",]=1
df.loc[df["Status"]=='spam',"Status",]=0
#print(df.head())

df_x=df["Message"]
df_y=df["Status"]

cv = CountVectorizer()

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

x_traincv = cv.fit_transform(["Hi How are you How are you doing","Hi what's up","Wow that's awesome"])
x_traincv = x_traincv.toarray()

#print(type(x_traincv))

import collections

a = ["my new car is my", "hi hi how are you?"]
#returns frequency of each word
corpus = collections.Counter(a)

#convert counter object to dictionary
corpus = dict(corpus)
print("Corpus:",corpus)