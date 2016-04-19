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

from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score

import re
import string
from random import shuffle
from numpy import array
import sys

path = "/Users/XinheLovesMom/Google Drive/data/training/"
testpath = "/Users/XinheLovesMom/Google Drive/data/training/"
liwcpath = path + "LIWC.csv"
liwctb = pd.read_csv(liwcpath)

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
agetb.columns = ["userId","agegroup"]

age_liwc = agetb.merge(liwctb,on="userId")
ageids = age_liwc["userId"]
ageliwc = age_liwc.iloc[:,1:]


def kfold(agetext,k,model,k2):
    import collections
    out = []
    for i in range(k):
        print "iteration: "+str(i)
        agetext = shuffle(agetext)
        datatb = agetext.iloc[:,1:]
        label = agetext["agegroup"].tolist()
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            datatb, label, test_size=0.15, random_state=i*6)
        data = X_train.values
        counter = collections.Counter(y_train)
        print counter
        testdata = X_test.values
        lsa = TruncatedSVD(k2, algorithm = 'arpack')
        normalizer = Normalizer(copy=False)
        X = lsa.fit_transform(data)
        X = normalizer.fit_transform(X)
        X_test = lsa.transform(testdata)
        X_test = normalizer.transform(X_test)
        model.fit(X,y_train)
        pred = model.predict(X_test)
        counter = collections.Counter(y_test)
        print counter
        counter = collections.Counter(pred)
        print counter
        out.append(round(accuracy_score(y_test, pred),5))
    print str(out)
    print np.mean(out)

# from sklearn.naive_bayes import MultinomialNB
# gnb = MultinomialNB(alpha=0.01)

def kfold2(agetext,k,model,k2):
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif
    import collections
    out = []
    for i in range(k):
        print "iteration: "+str(i)
        agetext = shuffle(agetext)
        datatb = agetext.iloc[:,1:]
        label = agetext["agegroup"].tolist()
        X_train, X_test, tlabel, testl = cross_validation.train_test_split(
            datatb, label, test_size=0.15, random_state=i*6)
        data = X_train.values
        counter = collections.Counter(y_train)
        print counter
        testdata = X_test.values
        selector = SelectKBest(f_classif,k=k2)
        X = selector.fit_transform(data,tlabel)
        X_test = selector.transform(testdata)
        model.fit(X,tlabel)
        pred = model.predict(X_test)
        counter = collections.Counter(testl)
        print counter
        counter = collections.Counter(pred)
        print counter
        out.append(round(accuracy_score(testl, pred),5))
    print str(out)
    print np.mean(out)

from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
clf = OneVsRestClassifier(SVC(kernel='linear'))
clf2 = OneVsOneClassifier(SVC(kernel='linear'))

# kfold(ageliwc,5,clf,10) 60%

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=5)
kfold2(ageliwc,5,clf,80)
