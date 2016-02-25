from os import listdir
from os.path import isfile,join
import csv
import pandas as pd
import numpy as np 

from nltk.stem import PorterStemmer
from nltk.tokenize import SpaceTokenizer
from nltk.corpus import stopwords
from functools import partial

import re
import string
from random import shuffle
from numpy import array

textpath = "/Users/XinheLovesMom/Google Drive/TCSS555/Train/Text"
textfiles = [f for f in listdir(textpath) if isfile(join(textpath, f))]
uids = [n.replace(".txt","") for n in textfiles]


def extText(textfiles):
	texttb = {}
	for tf in textfiles:
		uid = tf.replace(".txt","")
		f = open(textpath+"/"+tf,'r',encoding='latin-1')
		txt = f.read()
		texttb[uid] = txt
		f.close()
	return texttb

texttb = extText(textfiles)

mypath = textpath.replace("Text","Profile/")+"Profile.csv"
o = open(mypath,'rU')
profiletb = csv.DictReader(o)

proftb = {}
for row in profiletb:
    proftb[row.get("userid")] = [row.get("gender"),row.get("age")]

profl = [(key,value[0],value[1]) for key,value in proftb.items()]
genders = [(key,value[0]) for key,value in proftb]

texttbl = [(key,value) for key,value in texttb.items()]

gendertb = pd.DataFrame(genders)
gendertb.columns = ["usrid","gender"]

### tokenize functions: 
def format1(label,m):
    puncs = string.punctuation.replace("!","").replace("?","")
    STOPWORDS = set(stopwords.words('english'))
    ## split into words
    words = re.split(' |,|;|//n|//',m)
    ## remove slashes
    words = [w.strip('\\') for w in words]
    words = [w.strip('//') for w in words]
    words = [w.replace("/","") for w in words]
    ## lower case
    words = [w.lower() for w in words]
    ## remove punc except ? and !
    words = [w for w in words if not w in puncs]
    ## remove stop words
    words = [w for w in words if not w in STOPWORDS]
    return (label,words)

texttbl2 = [format1(key,value) for key,value in texttbl]

def format2(label,words):
    stemmed = [w.rstrip('.') for w in words]
    words = []
    for w in stemmed:
        ## check if the word is only contains ! or ?
        regexp = re.compile("^[?]+$|^[!]+$")
        if re.search(regexp,w) is not None:
            for s in list(w):
               words.append(s)
        # if the word ends with many question mark or !
        elif w.endswith(('!','?')):
            l = re.findall('[?!.]',w)
            words + l
        ## if the word only contains the \w leave it
        elif re.match(r'^[_\W]+$',w):
            words.append(w)
        else:
            w = re.sub(r'[?!.]','',w)
            words.append(w)
    return (label,words)

texttbl3 = [format2(key,value) for key,value in texttbl2]
    
def format3(label,words):
    ## split by the \\\n
    STEMMER = PorterStemmer()
    wl = []
    for w in words:
      wl = wl + w.split('\\\n')
    words = []
    for w in wl:
      words = words + w.split('\\')
    wl = []
    for w in words:
      wl = wl + w.split('/')
    words = wl
    words = [w.strip('\\') for w in words]
    words = [w.strip('//') for w in words]
    words = [w.replace("/","") for w in words]
    ## remove all the numbers:
    wl = []
    for w in words:
      if w.isdigit():
      	wl.append("_NUMBERS_")
      else:
        wl.append(''.join(i for i in w if not i.isdigit()))
    ## removing stopping words
    STOPWORDS = set(stopwords.words('english'))
    words = [w for w in wl if not w in STOPWORDS]
    stemmed = [STEMMER.stem(w) for w in words]
    ## if the word has length greater than 
    words = []
    for w in stemmed:
        if len(w) >= 12 and len(w) < 17:
         words.append("12_LONG_WORD")
        elif len(w) >= 17:
         words.append("17_LONG_WORD")
        else:
         words.append(w)
    return (label,words)

texttbl4 = [format3(key,value) for key,value in texttbl3]

###### gender #######
shuffle(texttbl4)
texttb = pd.DataFrame(texttbl4)
texttb.columns = ["usrid","text"]

# convert to panda dataframe:
gender_text = gendertb.merge(texttb,on="usrid")
## table gender with raw text
gendertext = gender_text[["gender","text"]]

# ## merge age and text tables
# age_text = agetb.merge(texttb,on="usrid")
# agetext = age_text[["age","text"]]

################## gender classification #####################
### fit model:
import numpy as np
from nltk.probability import FreqDist
from nltk.classify import SklearnClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

## data preparation
X = gendertext["text"]
X = X.tolist()
label = gendertext["gender"].tolist()
## accumulate all terms 
docs = []
for doc in X:
    docs.append(" ".join(doc))


## tokenize function used in vectorizer:
def tokenize(text):
    import re
    from string import punctuation
    r = re.compile(r'[\s{}]+'.format(re.escape(punctuation)))
    tokens = text.split(" ")
    tokens = [token.replace("\n","") for token in tokens]
    t = []
    for s in tokens:
        if s != "_NUMBERS_" and s!= "12_LONG_WORD" and s!= "17_LONG_WORD":
            if not re.match(r'^[_\W]+$', s):
                t = t + r.split(s)
            else:
                t.append(s)
        else:
            t.append(s)
    ## if the word has length greater than 
    words = []
    for w in t:
        if len(w) >= 12 and len(w) < 17:
         words.append("12_LONG_WORD")
        elif len(w) >= 17:
         words.append("17_LONG_WORD")
        else:
         words.append(w)
    tokens = []
    for w in words:
        if w=="!" or w=="?" or len(w)>1:
            tokens.append(w)
    return tokens

## 80000 most frequentiest terms
vec = TfidfVectorizer(encoding="latin-1",tokenizer = tokenize,token_pattern=r'(?u)\b\w\w+\b|^[_\W]+$',lowercase=False,max_features=50000)
# remove the spaces
docs2 = [doc.replace("\t","").replace("\n","") for doc in docs]
X = vec.fit_transform(docs2)
## chi2 method to select best 5000 terms
selector = SelectKBest(chi2,k=5000)
allvec_new = selector.fit_transform(X.toarray(),label)

### gaussian naive bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
# gnb.fit(allvec_new[:6650], label[:6650])
# pred = gnb.predict(allvec_new[6650:])

# print(np.mean(pred == label[6650:]))
# # 0.78877192982456146
# ## select 8000 terms
# selector = SelectKBest(chi2,k=8000)
# allvec_new = selector.fit_transform(X.toarray(),label)
# gnb.fit(allvec_new[:6650], label[:6650])
# pred = gnb.predict(allvec_new[6650:])
# print(np.mean(pred == label[6650:]))
# # 0.804210526316
# ## select 10000 'best' terms
# selector = SelectKBest(chi2,k=10000)
# allvec_new = selector.fit_transform(X.toarray(),label)
# gnb.fit(allvec_new[:6650], label[:6650])
# pred = gnb.predict(allvec_new[6650:])
# print(np.mean(pred == label[6650:]))
# # 0.821052631579

##### 5 folds cross validation #####
def genderClassify(model,label,top,X):
    selector = SelectKBest(chi2,k=top)
    allvec_new = selector.fit_transform(X.toarray(),label)
    scores = cross_validation.cross_val_score(model, allvec_new, label, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print(str(scores.tolist()))

### 5 folds cv of gaussian naive bayes 
genderClassify(gnb,label,10000,X)
# Accuracy: 0.83 (+/- 0.01)  *** very satisfied haha
# [0.8311415044713308, 0.8327196212519726, 0.8221052631578948, 0.8183254344391785, 0.8251711427066877]
## no difference when keep increase the #of selected terms in dtm
# >>> genderClassify(gnb,label,12000,X)
# Accuracy: 0.83 (+/- 0.01)
# [0.8348237769594951, 0.8342977380326144, 0.8242105263157895, 0.8204318062137967, 0.8293838862559242]
# >>> genderClassify(gnb,label,15000,X)
# Accuracy: 0.83 (+/- 0.01)
# [0.8374539715938979, 0.8390320883745397, 0.8236842105263158, 0.8230647709320695, 0.832016850974197]


############## text mining classify age group ######################
mypath = textpath.replace("Text","Profile/")+"Profile.csv"
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
    age = int(age)
    if age >= 18 and age <= 24:
        return "18-24"
    elif age >= 25 and age <= 34:
        return "25-34"
    elif age >= 35 and age <= 49:
        return "35-49"
    else:
        return "50-xx"

####
agegroups = [groupAge(age) for age in ages]

agetb["age"] = agegroups
agetb.columns = ["usrid","agegroup"]
label = agetext["agegroup"].tolist()

### multiple labels learning:
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm

## function
def AgeClassifier(freq1,best2,docs,label,model):
  vec = TfidfVectorizer(encoding="latin-1",tokenizer = tokenize,token_pattern=r'(?u)\b\w\w+\b|^[_\W]+$',lowercase=False,max_features=freq1)
  selector = SelectKBest(chi2,k=best2)
  X = vec.fit_transform(docs)
  allvec_new = selector.fit_transform(X.toarray(),label)
  scores = cross_validation.cross_val_score(model, allvec_new, label, cv=5)
  print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

### after several experiments linear svc gave the best result:
clf = OneVsRestClassifier(LinearSVC(random_state=0))
# If the number of features is much greater than the number of samples, the method is likely to give poor performances.
# SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).
# less features supposed to give better results: since we only have 6650 samples
# AgeClassifier(20000,10000,docs2,label,clf)
# Accuracy: 0.63 (+/- 0.02)
# AgeClassifier(50000,5000,docs2,label,clf)  ## best for now
# Accuracy: 0.64 (+/- 0.01)
# AgeClassifier(50000,20000,docs2,label,clf)
# Accuracy: 0.63 (+/- 0.02)
# AgeClassifier(50000,12000,docs2,label,clf)
# Accuracy: 0.64 (+/- 0.02)
# AgeClassifier(50000,4000,docs2,label,clf)
# Accuracy: 0.62 (+/- 0.01)
# AgeClassifier(50000,5000,docs2,label,clf)
# Accuracy: 0.63 (+/- 0.01)
AgeClassifier(50000,8000,docs2,label,clf)
# Accuracy: 0.64 (+/- 0.01)

### gaussian naive bayes:
# AgeClassifier(50000,8000,docs2,label,gnb)
# # Accuracy: 0.61 (+/- 0.02)
# AgeClassifier(50000,5000,docs2,label,gnb)
# # Accuracy: 0.61 (+/- 0.01)
# AgeClassifier(50000,12000,docs2,label,gnb)
# Accuracy: 0.59 (+/- 0.02)
