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
mypath = path + "profile/profile.csv"


liwcpath = path + "LIWC.csv"
liwctb = pd.read_csv(liwcpath)

o = open(mypath,'rU')
profiletb = csv.DictReader(o)

pertb = {}
for row in profiletb:
    pertb[row.get("userid")] = [row.get("ope"),row.get("con"),row.get("ext"),row.get("agr"),row.get("neu")]

perl = [(key,value[0],value[1],value[2],value[3],value[4]) for key,value in pertb.items()]
pertb = pd.DataFrame(perl)
pertb.columns = ["userId","ope","con","ext","agr","neu"]
person_text = pertb.merge(liwctb,on="userId")

# ## function to help with selecting the best features
def f_regression(X,Y):
   import sklearn
   return sklearn.feature_selection.f_regression(X,Y,center=False) 

# def getTextVecTest(best1,train_tb,testtb):
#     ## vec and tokenize function
#     vec = TfidfVectorizer(encoding="latin-1",tokenizer = tokenize,token_pattern=r'(?u)\b\w\w+\b|^[_\W]+$',lowercase=False,max_features=best1)
#     X = vec.fit_transform(docs2)
#     X_test = vec.transform(testdocs)
#     return (X,X_test)


from sklearn import linear_model
from sklearn.metrics import mean_squared_error
regression = linear_model.LinearRegression()
userids = person_text["userId"]
# get rid of the userid column
persontext = person_text.iloc[:,1:]

## testing and training:

texttb = persontext.iloc[:,5:]
traintb = texttb[:7999]
testtb = texttb[8000:9500] 
from sklearn.decomposition import TruncatedSVD

for i in range(5):
  colname = persontext.columns[i]
  print "personality: "+colname 
  label = persontext[colname]
  real = [float(n) for n in label[:7999].tolist()]
  testl = [float(n) for n in label[8000:9500].tolist()]
  # selector = SelectKBest(f_regression,20)
  # data = selector.fit_transform(traintb.values,real)
  # testdata = selector.transform(testtb.values)
  # real = [float(n) for n in [colname]]
  regression.fit(traintb.values,real)
  pred = regression.predict(testtb.values).tolist()
  # testl = [float(n) for n in testtb[colname]]
  print(mean_squared_error(testl, pred)**0.5)
  # dp2[colname] = pred.tolist()
