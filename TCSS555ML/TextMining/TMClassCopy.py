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

textpath = "/Users/XinheLovesMom/Google Drive/data/training/text/"
textfiles = [f for f in listdir(textpath) if isfile(join(textpath, f))]
uids = [n.replace(".txt","") for n in textfiles]

import io
def extText(textfiles,textpath):
	texttb = {}
	for tf in textfiles:
		uid = tf.replace(".txt","")
		f = io.open(textpath+"/"+tf,'r',encoding='latin-1')
		txt = f.read()
		texttb[uid] = txt
		f.close()
	return texttb

texttb = extText(textfiles,textpath)
texttbl = [(key,value) for key,value in texttb.items()]
# testtbl = [(key,value) for key,value in testtb.items()]

#####################################################
mypath = textpath.replace("text","profile")+"profile.csv"
o = open(mypath,'rU')
profiletb = csv.DictReader(o)

proftb = {}
for row in profiletb:
    proftb[row.get("userid")] = [row.get("gender"),row.get("age")]

profl = [(key,value[0],value[1]) for key,value in proftb.items()]
genders = [(value[0],value[1]) for value in profl]

#################
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
# testtbl2 = [format1(key,value) for key,value in testtbl]

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
# testtbl3 = [format2(key,value) for key,value in testtbl2]

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
# testtbl4 = [format3(key,value) for key,value in testtbl3]

###### gender #######
shuffle(texttbl4)
texttb = pd.DataFrame(texttbl4)
texttb.columns = ["usrid","text"]

# convert to panda dataframe:
gender_text = gendertb.merge(texttb,on="usrid")
## table gender with raw text
gendertext = gender_text[["gender","text"]]

#########################################   
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

## data preparation
X = gendertext["text"]
X = X.tolist()
label = gendertext["gender"].tolist()
## accumulate all terms 
docs = []
for doc in X:
    docs.append(" ".join(doc))

# remove the spaces
docs2 = [doc.replace("\t","").replace("\n","") for doc in docs]

#******************************************TRAINING AND TESTING SEPERATE*****************************************************
traindocs = docs2[:7999]
testdocs = docs2[8000:9500]
validatedocs = docs2[8000:9500]
### feature selection:
# vec = TfidfVectorizer(encoding="latin-1",tokenizer = tokenize,token_pattern=r'(?u)\b\w\w+\b|^[_\W]+$',lowercase=False,max_features=50000,use_idf=True)

# selector = SelectKBest(chi2,k=100)

label = gendertext["gender"].tolist()
tlabel = label[:7999]
testl = label[8000:9500]
# allvec_new = selector.fit_transform(X.toarray(),tlabel)


#########################################   

### gaussian naive bayes
# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# from sklearn.naive_bayes import MultinomialNB
# clf = MultinomialNB()
# ### fitting the model:
# clf.fit(allvec_new, tlabel)
# gnb.fit(allvec_new, tlabel)
# X_test = vec.transform(testdocs)
# X_test = selector.transform(X_test)
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
# y_pred = clf.predict(X_test.toarray()).tolist()
# accuracy_score(testl, y_pred)


from sklearn import linear_model

# clf = linear_model.SGDClassifier()

# X = vec.fit_transform(traindocs)

# for i in range(20,200,20):
# 	clf = linear_model.SGDClassifier()
# 	pca = TruncatedSVD(n_components=i)
# 	data = pca.fit_transform(X)
# 	X_test = vec.transform(testdocs)
# 	testdata = pca.transform(X_test)
# 	clf.fit(data, tlabel)
# 	y_pred = clf.predict(testdata).tolist()
# 	out = accuracy_score(testl, y_pred)
# 	print "accuracy for pca="+str(i)+" is: "+ str(out)

# pca = TruncatedSVD(n_components=160)
# # pca = TruncatedSVD(n_components=200)
# data = pca.fit_transform(X)
# X_test = vec.transform(testdocs)
# testdata = pca.transform(X_test)
# clf.fit(data, tlabel)
# y_pred = clf.predict(testdata).tolist()
# print(accuracy_score(testl, y_pred))

print "experiment on normalize the matrix and use tfidf matrix"
from sklearn.pipeline import make_pipeline
from sklearn import svm

# vec = TfidfVectorizer(encoding="latin-1",tokenizer = tokenize,token_pattern=r'(?u)\b\w\w+\b|^[_\W]+$',lowercase=False,max_features=15000,min_df=2,use_idf=True,ngram_range=(1,3))
# print vec
# X = vec.fit_transform(traindocs)


# ## feature selection

# pca = TruncatedSVD(n_components=180)
# normalizer = Normalizer(copy=False)
# lsa = make_pipeline(pca,normalizer)
# data = lsa.fit_transform(X)
# # ## prepare testing dataset: ##
# X_test = vec.transform(testdocs)
# testdata = lsa.transform(X_test)
# explained_variance = pca.explained_variance_ratio_.sum()
# print "Explained variance of the SVD step: {}%".format(int(explained_variance * 100))
# clf = linear_model.SGDClassifier().fit(data,tlabel)
# print "fitting the model:,,, svc"
# y_pred = clf.predict(testdata).tolist()
# print(accuracy_score(testl, y_pred))

# print "fitting the linearsvc model:"
# from sklearn import svm
# clf = svm.LinearSVC().fit(data,tlabel)
# clf2 = svm.SVC().fit(data,tlabel)

# y_pred = clf.predict(testdata).tolist()

# y_pred2 = clf2.predict(testdata).tolist()
# print(accuracy_score(testl, y_pred))
# print(accuracy_score(testl, y_pred2))


mypath = textpath.replace("text","profile/")+"profile.csv"
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
agetb.columns = ["usrid","agegroup"]
age_text = agetb.merge(texttb,on="usrid")
agetext = age_text[["agegroup","text"]]
# X = agetext["text"]
# X = X.tolist()
# label = agetext["agegroup"].tolist()
# tlabel = label[:7999]
# testl = label[8000:9500]

from sklearn.multiclass import OneVsRestClassifier
# import numpy as np
# from sklearn import datasets

# clf = OneVsRestClassifier(svm.LinearSVC()).fit(data,tlabel)
# y_pred = clf.predict(testdata).tolist()
# print(accuracy_score(testl, y_pred))
# writer=csv.writer(open("output.csv",'wb'))
# for n_components in range(100,210,20):
#     for nfeatures in range(5000,54000,5000):
#         print "numbers of components: "+str(n_components)
#         print "numbers of features:"+str(nfeatures)
#         vec = TfidfVectorizer(encoding="latin-1",tokenizer = tokenize,token_pattern=r'(?u)\b\w\w+\b|^[_\W]+$',lowercase=False,max_features=nfeatures,max_df=0.7,min_df=2,use_idf=True,ngram_range=(1,2))
#         X = vec.fit_transform(traindocs)
#         pca = TruncatedSVD(n_components=n_components)
#         normalizer = Normalizer(copy=False)
#         lsa = make_pipeline(pca,normalizer)
#         data = lsa.fit_transform(X)
#         X_test = vec.transform(testdocs)
#         testdata = lsa.transform(X_test)
#         clf = svm.LinearSVC().fit(data,tlabel)
#         y_pred = clf.predict(testdata).tolist()
#         out = accuracy_score(testl, y_pred)
#         print out
#         writer.writerow([n_components,nfeatures,out])

#### k folds cross validation: #####

def shuffle(df, n=1, axis=0):     
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df

def kfold(agetext,k,model,nfeatures,check=False,k2 = None,max_df=0.9,min_df=3):
    out = []
    for i in range(k):
        print "iteration: "+str(i)
        agetext = shuffle(agetext)
        X = agetext["text"]
        X = X.tolist()
        label = agetext["agegroup"].tolist()
        vec = TfidfVectorizer(tokenizer = tokenize,token_pattern=r'(?u)\b\w\w+\b|^[_\W]+$',lowercase=False,max_features=nfeatures,max_df = max_df,min_df = min_df,use_idf=True,ngram_range=(1,2))
        docs = []
        for doc in X:
            docs.append(" ".join(doc))
        docs2 = [doc.replace("\t","").replace("\n","") for doc in docs]
        traindocs = docs2[:7999]
        X = vec.fit_transform(traindocs)
        testdocs = docs2[8000:9500]
        X_test = vec.transform(testdocs)
        tlabel = label[:7999]
        testl = label[8000:9500]
        if(check):
            lsa = TruncatedSVD(k2, algorithm = 'arpack')
            normalizer = Normalizer(copy=False)
            X = lsa.fit_transform(X)
            X = normalizer.fit_transform(X)
            X_test = lsa.transform(X_test)
            X_test = normalizer.transform(X_test)
        model.fit(X,tlabel)
        pred = model.predict(X_test)
        out.append(round(accuracy_score(testl, pred),2))
    print str(out)
    print np.mean(out)


        
from sklearn.naive_bayes import MultinomialNB
gnb = MultinomialNB(alpha=0.01)
# kfold(agetext,5,gnb,20000)
# clf = linear_model.SGDClassifier()
# kfold(agetext,5,clf,20000)
# kfold(agetext,5,clf,20000)
# kfold(agetext,5,gnb,5000)
# [0.58733333333333337, 0.59933333333333338, 0.59199999999999997, 0.59799999999999998, 0.58266666666666667]
# kfold(agetext,5,gnb,10000)
# # [0.59466666666666668, 0.57199999999999995, 0.59399999999999997, 0.59399999999999997, 0.59599999999999997]
# kfold(agetext,5,gnb,4000)
# [0.59933333333333338, 0.59533333333333338, 0.61066666666666669, 0.58133333333333337, 0.61199999999999999]
# kfold(agetext,5,gnb,50000) worst
# kfold(agetext,5,gnb,50000,True,50)
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# clf = SGDClassifier(alpha=0.00001)
# kfold(agetext,5,clf,10000,True,100)
# kfold(agetext,5,gnb,5000)

print "grid.....searching...."

# parameters = {
#     # 'vect__max_df': (0.5, 0.75, 1.0),
#     # #'vect__max_features': (None, 5000, 10000, 50000),
#     # 'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
#     # 'tfidf__use_idf': (True, False),
#     #'tfidf__norm': ('l1', 'l2'),
#     'clf__alpha': (0.00001, 0.000001),
#     'clf__penalty': ('l2', 'elasticnet'),
#     #'clf__n_iter': (10, 50, 80),
# }
# kfold(agetext,5,clf,10000)
# kfold(agetext,5,gnb,5000)
# # [0.57866666666666666, 0.58999999999999997, 0.59866666666666668, 0.58133333333333337, 0.58533333333333337]
# kfold(agetext,5,gnb,3000) 
kfold(agetext,5,gnb,1000) 

# [0.59933333333333338, 0.57199999999999995, 0.59666666666666668, 0.57733333333333337, 0.59866666666666668]
# [0.61199999999999999, 0.57933333333333337, 0.58666666666666667, 0.60466666666666669, 0.62133333333333329]
# print "50000"
# kfold(agetext,5,gnb,50000)
# # [0.57599999999999996, 0.57599999999999996, 0.58733333333333337, 0.58133333333333337, 0.58199999999999996]
# print "1000"
# kfold(agetext,5,gnb,100000)
# [0.58066666666666666, 0.58466666666666667, 0.61066666666666669, 0.59133333333333338, 0.60866666666666669]
from sklearn.svm import SVC
clf = OneVsRestClassifier(SVC(kernel='linear'))
from sklearn.multiclass import OneVsOneClassifier
clf2 = OneVsOneClassifier(SVC(kernel='linear'))
# kfold(agetext,10,clf,3000)
# kfold(agetext,5,clf,50000,check=True,k2=100)
# [0.59066666666666667, 0.59799999999999998, 0.58933333333333338, 0.61066666666666669, 0.60733333333333328]
# kfold(agetext,5,clf,50000,check=True,k2=50)
# [0.60999999999999999, 0.59333333333333338, 0.60466666666666669, 0.59866666666666668, 0.58466666666666667]
# kfold(agetext,10,clf,50000,check=True,k2=20)
# [0.61, 0.61, 0.58, 0.6, 0.6]
# kfold(agetext,3,clf2,50000,check=True,k2=20)
# [0.60799999999999998, 0.60866666666666669, 0.61733333333333329, 0.61333333333333329, 0.59066666666666667]
# from sklearn.neighbors import KNeighborsClassifier
# neigh = KNeighborsClassifier(n_neighbors=15)
# kfold(agetext,3,neigh,50000,check=True,k2=20)

# kfold(agetext,5,clf,80000,check=True,k2=20)
# [0.58, 0.6, 0.62, 0.59, 0.59]
# [0.61933333333333329, 0.61466666666666669, 0.58799999999999997, 0.61266666666666669, 0.59733333333333338]
# kfold(agetext,5,clf,15000,check=True,k2=20)
# [0.59, 0.6, 0.58, 0.6, 0.61]
# kfold(agetext,5,clf,80000,check=True,k2=30)

def kfold2(agetext,k,model,nfeatures,check=False,k2 = None,max_df=0.75,min_df=2):
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    out = []
    for i in range(k):
        print "iteration: "+str(i)
        agetext = shuffle(agetext)
        X = agetext["text"]
        X = X.tolist()
        label = agetext["agegroup"].tolist()
        vec = TfidfVectorizer(tokenizer = tokenize,token_pattern=r'(?u)\b\w\w+\b|^[_\W]+$',lowercase=False,max_features=nfeatures,max_df = max_df,min_df = min_df,use_idf=True,ngram_range=(1,2))
        docs = []
        for doc in X:
            docs.append(" ".join(doc))
        docs2 = [doc.replace("\t","").replace("\n","") for doc in docs]
        traindocs = docs2[:7999]
        X = vec.fit_transform(traindocs)
        testdocs = docs2[8000:9500]
        X_test = vec.transform(testdocs)
        tlabel = label[:7999]
        testl = label[8000:9500]
        if(check):
            selector = SelectKBest(chi2,k=k2)
            X = selector.fit_transform(X,tlabel)
            X_test = selector.transform(X_test)
        model.fit(X,tlabel)
        pred = model.predict(X_test)
        out.append(round(accuracy_score(testl, pred),2))
    print str(out)
    print np.mean(out)

# kfold2(agetext,10,gnb,50000)
# for i in range(5000,26000,3000):
#     kfold2(agetext,5,gnb,i)

# kfold2(agetext,5,gnb,10000,True,1000)

# 0.602
# kfold2(agetext,5,gnb,10000,True,2000) #6.06
# kfold2(agetext,5,gnb,None,True,10000) #59%
# kfold2(agetext,5,gnb,None,True,500) 60%
# kfold2(agetext,5,gnb,None,True,20000) 56%
# kfold2(agetext,5,gnb,30000,True,2000) #0.602
# kfold2(agetext,5,gnb,10000,True,3000) 59%
# kfold2(agetext,5,gnb,20000)
# kfold2(agetext,5,gnb,30000,True,1000)
# kfold2(agetext,5,gnb,30000,True,5000)
# kfold2(agetext,5,gnb,10000,True,5000) below 60..
# kfold2(agetext,5,gnb,10000,True,5000) #59%
# kfold2(agetext,5,gnb,50000) #59%
# kfold2(agetext,5,gnb,100000,True,1000)
# kfold(agetext,5,clf,5000,True,k2=20) #0.606
# from sklearn.neighbors import KNeighborsClassifier
# kfold(agetext,5,gnb,10000) #0.9,3
# [0.6, 0.59, 0.59, 0.59, 0.61] 10000
# 0.596
# [0.59, 0.6, 0.59, 0.6, 0.58] 5000
# 0.592
# kfold(agetext,3,clf,10000)
# kfold(agetext,3,clf,5000,True,k2=10)