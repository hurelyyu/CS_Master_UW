#!/usr/bin/python2.7
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
from numpy import array
import sys

###import package for nonface-image
import cv2
import random
import math
from time import time 
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA
from sklearn.feature_extraction import image
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import os
from PIL import Image

### reading test files
try:
    testpath = sys.argv[2]
    outputpath = sys.argv[4]
    testpath = testpath
    print "Test Data is at " + testpath
    print "Output Folder is: " + outputpath
except IndexError as e:
    print "ERROR: input paths are required: 1: path to test data 2: output path"
    sys.exit()

############################################################################################
######################### gender classification from images oxford.csv #####################
############################################################################################
path = "/data/training/"
#testpath = "/data/public-test-data/"
testtextpath = testpath + "text/"
textpath = "/data/training/text/"
textfiles = [f for f in listdir(textpath) if isfile(join(textpath, f))]
############## test dataset ##################
testfiles = [f for f in listdir(testtextpath) if isfile(join(testtextpath, f))]
#############################################
## get the usr ids 
uids = [n.replace(".txt","") for n in textfiles]
testuids = [n.replace(".txt","") for n in testfiles]

## read the oxford csv file 
facetb = pd.read_csv(path + "oxford.csv")
testoxtb = pd.read_csv(testpath + "/oxford.csv")

## read gender and agegroup
profile_path = path + "profile/profile.csv"
o = open(profile_path,'rU')
profiletb = csv.DictReader(o)

proftb = {}
for row in profiletb:
    proftb[row.get("userid")] = [row.get("gender"),row.get("age")]

profl = [(key,value[0],value[1]) for key,value in proftb.items()]
genders = [(value[0],value[1]) for value in profl]

gendertb = pd.DataFrame(genders)
gendertb.columns = ["userId","gender"]
# convert to panda dataframe:
gender_image = gendertb.merge(facetb,on="userId")
gender_image.drop(['userId','faceID'], axis=1, inplace=True)

### data preparetion for gender classification ###

##### training data set: #####
label = gender_image["gender"].tolist()
X = gender_image.drop(['gender'], axis=1)
data = X.as_matrix(columns=X.columns[1:])

## fitting the model 13nn and top 3 best features
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2

# top 3 features for oxford get back the best result
selector = SelectKBest(f_classif, k=3)
X_new_train = selector.fit_transform(data, label)
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_new_train,label)
print "Model is Ready...Starting Testing..."

## userids in test oxford csv file
test_ox_usrids = testoxtb["userId"].tolist()
X_test = testoxtb.drop(['userId','faceID'], axis=1)

## prepare the testing data
testdata = X_test.as_matrix(columns=X_test.columns[1:])
testdata = selector.transform(testdata)
image_pred = knn.predict(testdata).tolist()
print "Image Gender Classification Finished for Oxford KNN Model"

###########################################################################################
############ code training all the image files and fit the test data without faces #######
###########################################################################################
print "starting fitting the test users without face detected"
## get the test users that is not in oxford csv file
## need to use opencv to test on the whole image
nofaceusrid_arr = [i for i in testuids if i not in test_ox_usrids]
print nofaceusrid_arr
#### get full image and resize as face information
faceCascade = cv2.CascadeClassifier('/home/ituser/Downloads/haarcascade_frontalface_alt.xmls/haarcascade_frontalface_alt.xml')
print faceCascade
gender_dic = {}
useri_arr = [] 
reader = csv.reader(open(path + "profile/profile.csv"))
for number , userid , age ,gender,ope , con , ext , agr , neu in reader:
    useri_arr.append(userid)
    gender_dic[userid]=gender

useri_arr = useri_arr[1:]
print len(useri_arr)
########################################################################
#Training Image Process
ip=0
value1 = []
output = []
Face = []
for key in range(0,len(useri_arr)):
    userid = useri_arr[key]
    userid
    value = gender_dic[userid]
    #print ip
    g = int(float(value)) 
    value1.append(g)
    image = Image.open(path + "image/"+userid+".jpg").convert('L')   
    #print image 
    image = np.array(image)
    image1 = cv2.resize(image,(30,30))
    image2 = image1.flatten()
    #print image2
    Face.append(image2)
    ip = ip+1

print ip
#print Face
print len(Face)
print "Training Face feature Extraction Finished" 
X = np.array(Face)
#print X[1]
n_features = X.shape[1]
print n_features
y = np.array(value1)
target_names = y
print len(y)
n_classes = target_names.shape[0] #19
print("Total dataset size:")
print("n_features: %d" % n_features) #900
print("n_classes: %d" % n_classes) #n_classes: 19
X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0, random_state=42)
n_components = X_train.shape[0]
t0 = time()
neigh = KNeighborsClassifier(n_neighbors=90)
clf = neigh.fit(X_train, y_train)
# print("Best estimator found by grid search:") #Best estimator found by grid search:
# print(clf.best_estimator_) 

########################################################################
#Test Image Process
nofaceusrid_arr
testuseri_arr = [] 
for userid in nofaceusrid_arr:
     testuseri_arr.append(userid)

ip=0
nofaceimg_arr = []
value2 = []
for key in range(0,len(testuseri_arr)):
    userid = testuseri_arr[key]
    print ip
    image = Image.open(testpath+"/image/"+userid+".jpg").convert('L')    
    image = np.array(image)
    image1 = cv2.resize(image,(30,30))
    image2 = image1.flatten()
    #print image2
    nofaceimg_arr.append(image2)
    ip = ip+1

########################################################################
# predict
X_test = np.array(nofaceimg_arr)
print("Predicting people's gender on the test set")
t0 = time()
image_pred2 = neigh.predict(X_test).tolist()
print("done in %0.3fs" % (time() - t0))   
#########################################################
testids = test_ox_usrids+nofaceusrid_arr
#print testids
imagepred = image_pred+image_pred2
#print imagepred
outdf = pd.DataFrame({"userid":testids,"gender":imagepred})
print len(outdf)
print "Gender Classification Process Finished"

#################################################################
############### age group classification using text #############
#################################################################
import numpy as np
from nltk.probability import FreqDist
from nltk.classify import SklearnClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


### data preparation: ####
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
testtb = extText(testfiles,testtextpath)
#####################################################
texttbl = [(key,value) for key,value in texttb.items()]
###### test #######
testtbl = [(key,value) for key,value in testtb.items()]
#################################################################################
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
###############################################################################
texttb = pd.DataFrame(texttbl4)
texttb.columns = ["usrid","text"]
testtb = pd.DataFrame(testtbl4)
testtb.columns = ["usrid","text"]

#############################################################################
from sklearn import linear_model
from sklearn.decomposition import TruncatedSVD
from sklearn import linear_model
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import numpy as np
from sklearn import datasets

############## text mining classify age group ######################
## age:
ages = [(value[0],value[2]) for value in profl]
agetb = pd.DataFrame(ages)
agetb.columns = ["userid","age"]

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
label = agetext["agegroup"].tolist()

## data preparation
X = agetext["text"]
X = X.tolist()
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

## 50000 most frequentiest terms 56%  30000 55% max_df=0.85
vec = TfidfVectorizer(encoding="latin-1",tokenizer = tokenize,token_pattern=r'(?u)\b\w\w+\b|^[_\W]+$',lowercase=False,max_features=80000,max_df=0.9,min_df=2,use_idf=True,ngram_range=(1,2))
# remove the spaces
docs2 = [doc.replace("\t","").replace("\n","") for doc in docs]
X = vec.fit_transform(docs2)

######## pca reduce dimensions #######
from sklearn.preprocessing import Normalizer
lsa = TruncatedSVD(20, algorithm = 'arpack')
normalizer = Normalizer(copy=False)
X = lsa.fit_transform(X)
data = normalizer.fit_transform(X)

### after several experiments linear svc gave the best result:
from sklearn.svm import SVC
clf = OneVsRestClassifier(SVC(kernel='linear'))
#### prepare training set and testing set:
#******************************************TRAINING AND TESTING SEPERATE*****************************************************
clf.fit(data, label)
print "finished fitting model for age-group classification"

#### prepare the test data text datasets #######
X_test = vec.transform(testdocs)
testdata = lsa.transform(X_test)
testdata = normalizer.transform(testdata)
### getting the predicted values: 
pred = clf.predict(testdata)
outdfage = pd.DataFrame({"userid": testuids, "agegroup": pred.tolist()})
#outdf["agegroup"] = pred.tolist()
outdf = pd.merge(outdf,outdfage,on="userid")

print "Gender and Age Group Classification Process finished"
#
####################################### personality ##############################
profile_path = path + "profile/profile.csv"
o = open(profile_path,'rU')
profiletb = csv.DictReader(o)

### read the texttb csv file to get the user ids and corresponding cleaned text data
# texttb = pd.read_csv("texttb.csv",encoding="latin-1")
## aggregate the data for personality
pertb = {}
for row in profiletb:
    pertb[row.get("userid")] = [row.get("ope"),row.get("con"),row.get("ext"),row.get("agr"),row.get("neu")]

perl = [(key,value[0],value[1],value[2],value[3],value[4]) for key,value in pertb.items()]
pertb = pd.DataFrame(perl)
pertb.columns = ["userid","ope","con","ext","agr","neu"]

## prepare data
liwcpath = path + "LIWC.csv"
testliwcpath = testpath + "LIWC.csv"
liwctb = pd.read_csv(liwcpath)
liwctb = liwctb.rename(columns={"userId":"userid"})
testliwctb = pd.read_csv(testliwcpath)
testliwctb = testliwctb.rename(columns={"userId":"userid"})

## training data prepared
person_text = pertb.merge(liwctb,on="userid")
## test data set ids:
userids = testliwctb["userid"]

##### use linear regression #####
from sklearn import linear_model
regression = linear_model.LinearRegression()
persontext = person_text.iloc[:,1:]
texttb = persontext.iloc[:,5:]
testtexttb = testliwctb.iloc[:,1:]


## accumulate the results
dp2 = pd.DataFrame({"userid":userids})

for i in range(5):
  colname = persontext.columns[i]
  print "personality: "+colname 
  real = [float(n) for n in persontext[colname]]
  regression.fit(texttb.values,real)
  pred = regression.predict(testtexttb.values)
  dp2[colname] = pred.tolist()

print "Linear Regression for Personality Prediction Process Finished!!" 
# dp2.to_csv(outputpath+"baysianRidge_regression_predicted_personality.csv.csv", cols = ("usrid","ope","con","ext","agr","neu"), encoding='utf-8',index=False)
# print "baysianRidge_regression_predicted_personality.csv file is generated." 

## merging two dataframe
dp3 = pd.merge(outdf,dp2,on="userid")

## remove duplicate
dp3 = dp3.drop_duplicates('userid', take_last=True)
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
