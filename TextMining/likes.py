#!/usr/local/bin/python2.7
from os import listdir
from os.path import isfile,join
import csv
import pandas as pd
import numpy as np 

from nltk.stem import PorterStemmer
from nltk.tokenize import SpaceTokenizer
from nltk.corpus import stopwords
from functools import partial
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import re
import string
from random import shuffle
from numpy import array
import sys

path = "/Users/XinheLovesMom/Google Drive/data/training/"
likepath = path + "Relation/Relation.csv"
likestb = pd.read_csv(likepath)
likestb = likestb.iloc[:,1:]
gb = likestb.groupby(("userid"))
result = gb["like_id"].unique()
# dataframe
likesdf = result.reset_index()

likesdf["like_id"] = likesdf["like_id"].apply(lambda x: " ".join(str(i) for i in x))



mypath = path + "profile/profile.csv"
o = open(mypath,'rU')
profiletb = csv.DictReader(o)

proftb = {}
for row in profiletb:
    proftb[row.get("userid")] = [row.get("gender"),row.get("age")]

proftb = [(key,value) for key,value in proftb.items()]
## age:
ages = [(key,value[1]) for key,value in proftb]
agetb = pd.DataFrame(ages)
agetb.columns = ["usrid","age"]

## change to age group:
ages = agetb["age"].tolist()
def groupAge(age):
    age = float(age)
    if age <= 24:
        return "xx-24"
    elif age >= 25 and age <= 34:
        return "25-34"
    elif age >= 35 and age <= 49:
        return "35-49"
    else:
        return "50-xx"

####
agegroups = [groupAge(age) for age in ages]

agetb["age"] = agegroups
agetb.columns = ["userid","agegroup"]
age_text = agetb.merge(likesdf,on="userid")
labels = age_text["agegroup"].tolist() 

corpus = likesdf["like_id"].tolist()
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1,analyzer = "word",tokenizer = None,stop_words = None,token_pattern = "\S+")
# len(vectorizer.get_feature_names())
# 536204
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()

X = vectorizer.fit_transform(corpus)
traindocs = X[:7999]
testdocs = X[8000:9500]

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# selector = SelectKBest(chi2, k=10000)
# X_new = selector.fit_transform(traindocs, labels[:7999])
# test_new = selector.transform(testdocs)
import collections
clf = MultinomialNB().fit(X_new, labels[:7999])
pred = clf.predict(test_new)
clf = MultinomialNB().fit(traindocs, labels[:7999])
clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42).fit(traindocs, labels[:7999])
pred = clf.predict(testdocs)
collections.Counter(pred)
collections.Counter(labels[8000:9500])

from sklearn.metrics import accuracy_score
accuracy_score(labels[8000:9500], pred)