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

import re
import string
from random import shuffle
from numpy import array
import sys

# textpath = "/Users/XinheLovesMom/Google Drive/TCSS555/Train/Text"
# outputpath = "/Users/XinhelovesMom/Google Drive/TCSS555/Public Test/Results/"

try:
    testpath = sys.argv[2]
    outputpath = sys.argv[4]
    print "Test Data is at " + testpath
    print "Output Folder is: " + outputpath
except IndexError as e:
    print "ERROR: input paths are required: 1: path to test data 2: output path"
    sys.exit()

textpath = "/data/training/text/"
textfiles = [f for f in listdir(textpath) if isfile(join(textpath, f))]
############## test dataset ##################
# testpath = "/Users/XinhelovesMom/Google Drive/TCSS555/Public Test/Text"
testfiles = [f for f in listdir(testpath) if isfile(join(testpath, f))]
#############################################
uids = [n.replace(".txt","") for n in textfiles]
testuids = [n.replace(".txt","") for n in testfiles]

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
################# testing dataset:########################
testtb = extText(testfiles,testpath)
#####################################################
mypath = textpath.replace("Text","Profile/")+"Profile.csv"
o = open(mypath,'rU')
profiletb = csv.DictReader(o)

proftb = {}
for row in profiletb:
    proftb[row.get("userid")] = [row.get("gender"),row.get("age")]

profl = [(key,value[0],value[1]) for key,value in proftb.items()]
genders = [(value[0],value[1]) for value in profl]

texttbl = [(key,value) for key,value in texttb.items()]
###### test #######
testtbl = [(key,value) for key,value in testtb.items()]
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
testtbl2 = [format1(key,value) for key,value in testtbl]

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
testtbl3 = [format2(key,value) for key,value in testtbl2]

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
testtbl4 = [format3(key,value) for key,value in testtbl3]

###### gender #######
shuffle(texttbl4)
texttb = pd.DataFrame(texttbl4)
texttb.columns = ["usrid","text"]
testtb = pd.DataFrame(testtbl4)
testtb.columns = ["usrid","text"]

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

###########test set docs ################
X_test = testtb["text"]
X_test = X_test.tolist()

testdocs = []
for doc in X_test:
    testdocs.append(" ".join(doc)) 

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
gnb.fit(allvec_new, label)
X_test = vec.transform(testdocs)
X_test = selector.transform(X_test)
################### test data get output and writeout files ###################
pred = gnb.predict(X_test.toarray())
outdf = pd.DataFrame({"userid":testuids,"gender":pred.tolist()})
####
# print("Gender is: "+str(pred.tolist()))
#####################################

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
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2) 
    print str(scores.tolist()) 

### 5 folds cv of gaussian naive bayes 
# genderClassify(gnb,label,10000,X)
# Accuracy: 0.83 (+/- 0.01)  *** best result so far
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
age_text = agetb.merge(texttb,on="usrid")
agetext = age_text[["agegroup","text"]]
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
  print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2) 

### after several experiments linear svc gave the best result:
clf = OneVsRestClassifier(LinearSVC(random_state=0))
#### prepare training set and testing set:
vec = TfidfVectorizer(encoding="latin-1",tokenizer = tokenize,token_pattern=r'(?u)\b\w\w+\b|^[_\W]+$',lowercase=False,max_features=50000)
selector = SelectKBest(chi2,k=5000)
X = vec.fit_transform(docs2)
## chi2 method to select best 5000 terms
allvec_new = selector.fit_transform(X.toarray(),label)
#### preparing the testing dataset 
X_test = vec.transform(testdocs)
X_test = selector.transform(X_test)
### fitting the model:
# clf.fit(allvec_new[:6650], label[:6650])
clf.fit(allvec_new, label)
### getting the predicted values: 
pred = clf.predict(X_test.toarray())
outdf["agegroup"] = pred.tolist()

print "Gender and Age Group Classification Process finished"

#### write the table to csv file contains gender and agegroup associated with userid
# outdf.to_csv(outputpath+"predicted_gender_agegroup.csv", cols = ("userid","gender","agegroup"), encoding='utf-8',index=False)
# print "predicted_gender_agegroup.csv is generated. Process finished"

# ## store the 
# texttb.to_csv("texttb.csv",cols=("userid","text"),index=False)
# print("texttb.csv file generated. stores the cleaned text data for each user")
## get the text cleaned docs
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
# AgeClassifier(50000,8000,docs2,label,clf) *** best
# Accuracy: 0.64 (+/- 0.01)

### gaussian naive bayes:
# AgeClassifier(50000,8000,docs2,label,gnb)
# # Accuracy: 0.61 (+/- 0.02)
# AgeClassifier(50000,5000,docs2,label,gnb)
# # Accuracy: 0.61 (+/- 0.01)
# AgeClassifier(50000,12000,docs2,label,gnb)
# Accuracy: 0.59 (+/- 0.02)


###################################### personality #########################
mypath = textpath.replace("Text","Profile/")+"Profile.csv"
o = open(mypath,'rU')
profiletb = csv.DictReader(o)

### read the texttb csv file to get the user ids and corresponding cleaned text data
# texttb = pd.read_csv("texttb.csv",encoding="latin-1")
## aggregate the data for personality
pertb = {}
for row in profiletb:
    pertb[row.get("userid")] = [row.get("ope"),row.get("con"),row.get("ext"),row.get("agr"),row.get("neu")]

perl = [(key,value[0],value[1],value[2],value[3],value[4]) for key,value in pertb.items()]
pertb = pd.DataFrame(perl)
pertb.columns = ["usrid","ope","con","ext","agr","neu"]
person_text = pertb.merge(texttb,on="usrid")

## function to help with selecting the best features
def f_regression(X,Y):
   import sklearn
   return sklearn.feature_selection.f_regression(X,Y,center=False) 

### get the vec
### inputs: train_tb and test_tb and 20000
def getTextVecTest(best1,train_tb,testtb):
    docs = train_tb["text"]
    textl = docs.tolist()
    X_test = testtb["text"]
    X_test = X_test.tolist()
    docs = []
    for doc in textl:
        docs.append(" ".join(doc))
    testdocs = []
    for doc in X_test:
        testdocs.append(" ".join(doc))
    docs2 = [doc.replace("\t","").replace("\n","") for doc in docs]
    ## vec and tokenize function
    vec = TfidfVectorizer(encoding="latin-1",tokenizer = tokenize,token_pattern=r'(?u)\b\w\w+\b|^[_\W]+$',lowercase=False,max_features=best1)
    X = vec.fit_transform(docs2)
    X_test = vec.transform(testdocs)
    return (X,X_test)

X,X_test = getTextVecTest(20000,person_text,testtb)

## fitting the model:
from sklearn import linear_model
from sklearn.decomposition import TruncatedSVD
regression = linear_model.LinearRegression()

## 50 dimensions from pca
pca = TruncatedSVD(n_components=50)
data = pca.fit_transform(X)
###########test set docs ################
# X_test = testtb["text"]
# X_test = X_test.tolist()

# testdocs = []
# for doc in X_test:
#     testdocs.append(" ".join(doc)) 

#########################################   
X_test = pca.transform(X_test)

## return pandas dataframe
# dp = pd.DataFrame({"userid":testuids})

# for i in range(5):
#   colname = person_text.columns[i+1]
#   print "personality: "+colname 
#   real = [float(n) for n in person_text[colname]]
#   regression.fit(data,real)
#   pred = regression.predict(X_test)
#   dp[colname] = pred.tolist()

# print "Linear Regression for Personality Prediction Process Finished!!" 
# # dp.to_csv(outputpath+"linear_regression_predicted_personality.csv", cols = ("usrid","ope","con","ext","agr","neu"), encoding='utf-8',index=False)
# print "linear_regression_predicted_personality.csv file is generated." 

##### use bayesian ridge regression #####
from sklearn import linear_model
clf = linear_model.BayesianRidge()
dp2 = pd.DataFrame({"userid":testuids})
for i in range(5):
  colname = person_text.columns[i+1]
  print "personality: "+colname 
  real = [float(n) for n in person_text[colname]]
  clf.fit(data,real)
  pred = clf.predict(X_test)
  dp2[colname] = pred.tolist()

print "Bayesian Ridge Regression for Personality Prediction Process Finished!!" 
# dp2.to_csv(outputpath+"baysianRidge_regression_predicted_personality.csv.csv", cols = ("usrid","ope","con","ext","agr","neu"), encoding='utf-8',index=False)
# print "baysianRidge_regression_predicted_personality.csv file is generated." 

## merging two dataframe
dp3 = pd.merge(outdf,dp2,on="userid")

## function to xml format
def toXML(row,filename,mode="w"):
    xml = ['<']
    # for field in row.index:
    xml.append('  userId="{0}"'.format(row["userid"]))
    xml.append('  age_group="{0}"'.format(row["agegroup"]))
    xml.append('  gender="{0}"'.format(row["gender"]))
    xml.append('  extrovert="{0}"'.format(row["ext"]))
    xml.append('  neurotic="{0}"'.format(row["neu"]))
    xml.append('  agreeable="{0}"'.format(row["agr"]))
    xml.append('  conscientious="{0}"'.format(row["con"]))
    xml.append('  open="{0}"'.format(row["ope"]))
    xml.append('/>')
    res = '\n'.join(xml)
    with open(filename,mode) as f:
        f.write(res)

### output the data as xml file for each user
print "writing out all the output xml files to " + outputpath + "..."
dp3.apply(lambda row: toXML(row, outputpath + row['userid'] + ".xml"), axis=1)

