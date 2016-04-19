import sys
import csv
import cv2
from time import time
import random
import numpy as np
from sklearn import cross_validation
from sklearn.feature_extraction import image
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline

import os
from PIL import Image
import pandas as pd

########################################################################
#Construct Matrix of age group from profile.csv
agematrix = [[] for x in xrange(9500)]
age_dic = {}
useri_arr = [] 

agedf = pd.read_csv('/Users/yaqunyu/UW_2016_Winter/TCSS555dataset/Train/Profile/Profile.csv',sep=',')
for index, row in agedf.iterrows():
    userid = row['userid']
    age = row['age']
    agematrix[index].append(userid)
    agematrix[index].append(str(age))

ageDF = pd.DataFrame(agematrix)
ageDF.columns=['userid','age']
ages = ageDF['age'].tolist()
agegroups = []
for age in ages:
	age = int(float(age))
	if age <= 24:
		agegroups.append("XX-24")
	elif age >= 25 and age <= 34:
		agegroups.append("25-34")
	elif age >= 35 and age <= 49:
		agegroups.append("35-49")
	else:
		agegroups.append("50-xx")

ageDF["age"]=agegroups
ageDF.columns = ["userid","agegroup"]
########################################################################
#Construct Matrix of all image information from oxford.csv
facetb = pd.read_csv('/Users/yaqunyu/UW_2016_Winter/oxford.csv')
ageDF.columns = ["userId","agegroup"]
age_image = pd.merge(ageDF,facetb,on="userId")
age_image.drop(['userId','faceID'], axis=1, inplace=True)
##### training data set: #####
label = age_image["agegroup"].tolist()
X = age_image.drop(['agegroup'], axis=1)
data = X.as_matrix(columns=X.columns[1:])

################################################################################################################################################
#Select k best from Sklearn module
################################################################################################################################################
# top k features for oxford get back the best result
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
selector = SelectKBest(f_classif, k=5)
X_new= selector.fit_transform(data, label)

########################################################################
## fitting the model knn and top k best features
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, Y_train, Y_test = train_test_split(X_new,label,test_size=0.3,random_state=0)
knn = KNeighborsClassifier(n_neighbors=70)
clf=knn.fit(X_train,Y_train)
n_samples = X_train.shape[0]
scores = cross_validation.cross_val_score(clf, X_train, Y_train, cv=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#Accuracy: 0.59 (+/- 0.00)

########################################################################
## fitting the model OneVsRestClassifier and top k best features using f_classif, feature number = 5
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
selector = SelectKBest(f_classif, k=5)
X_new= selector.fit_transform(data, label)
X_train, X_test, Y_train, Y_test = train_test_split(X_new,label,test_size=0.3,random_state=0)
classif = OneVsRestClassifier(SVC(kernel='linear'))
clf = classif.fit(X_train, Y_train)
n_samples = X_train.shape[0]
scores = cross_validation.cross_val_score(clf, X_train, Y_train, cv=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

########################################################################
## fitting the model OneVsRestClassifier and top k best features using chi2(no negative allowed), feature number = 5
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
selector = SelectKBest(chi2, k=5)
X_new_train = selector.fit_transform(data.clip(0), label)
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, Y_train, Y_test = train_test_split(X_new_train,label,test_size=0.3,random_state=0)
knn = KNeighborsClassifier(n_neighbors=30)
clf=knn.fit(X_train,Y_train)
n_samples = X_train.shape[0]
scores = cross_validation.cross_val_score(clf, X_train, Y_train, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#Accuracy: 0.59 (+/- 0.00) 3/7 kbest k=5 knn k =70 array([ 0.59009009,  0.5915239 ,  0.58881876,  0.59132007,  0.59132007])
#Accuracy: 0.60 (+/- 0.00) 2.5/7.5 
########################################################################
## fitting the model NeuralNetwork and top k best features
from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from NeuralNetwork import NeuralNetwork

selector = SelectKBest(f_classif, k=20)
X_new= selector.fit_transform(data, label)
X_train, X_test, Y_train, Y_test = train_test_split(X_new,label,test_size=0.3,random_state=0)
logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)
classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
rbm.learning_rate = 0.1
rbm.n_iter = 20

# More components tend to give better prediction performance, but larger
# fitting time

rbm.n_components = 100
logistic.C = 8000.0

# Training RBM-Logistic Pipeline

classifier.fit(X_train, Y_train)

# Training Logistic regression

logistic_classifier = linear_model.LogisticRegression(C=200.0)
clf2=logistic_classifier.fit(X_train, Y_train)

n_samples = X_train.shape[0]
scores = cross_validation.cross_val_score(clf2, X_train, Y_train, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# 0.59
###############################################################################
#SVD
from sklearn import linear_model
from sklearn.decomposition import TruncatedSVD
from sklearn import linear_model
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import numpy as np
from sklearn import datasets

n_components = 5 
pca = TruncatedSVD(n_components=n_components)
# data = pca.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X_new,label,test_size=0.3,random_state=0)
###after several experiments linear svc gave the best result:
clf = OneVsRestClassifier(LinearSVC(random_state=0))
clf2 = clf.fit(X_train, Y_train)
#cv = cross_validation.ShuffleSplit(n_samples, n_iter=5,test_size = 0.3, random_state = 0)
scores=cross_validation.cross_val_score(clf2, X_train, Y_train, cv=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# 54% array([ 0.47755544,  0.59859155,  0.53821138]) cv=3
# 0.42 (+/- 0.42) array([ 0.59009009,  0.59062218,  0.58972047,  0.09584087,  0.25497288]) cv=5
###############################################################################
#unsupervised SVM
from sklearn import svm
clf = svm.SVC()
data = X_new.as_matrix(columns=X_new.columns[1:])
label = age_image["agegroup"].tolist()
X_train, X_test, Y_train, Y_test = train_test_split(data,label,test_size=0.3,random_state=0)
clf2 = clf.fit(X_train, Y_train)
scores=cross_validation.cross_val_score(clf2, X_train, Y_train, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#non select key best, for all features svm 0.59 test_size=0.25, random_state=42 array([ 0.59046283,  0.59046283,  0.59046283,  0.59058989,  0.59100492])
###############################################################################
########################################################################
#Bernoulli Naive Bayes
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
X_train, X_test, Y_train, Y_test = train_test_split(X_new,label,test_size=0.3,random_state=42)
clf2 = clf.fit(X_train, Y_train)
n_samples = X_train.shape[0]
scores = cross_validation.cross_val_score(clf2, X_train, Y_train, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


################################################################################################################################################
#Construct Matrix of partial image information from oxford.csv by Maually Select features related close to age
################################################################################################################################################
oxfordprofil_path = '/Users/yaqunyu/UW_2016_Winter/oxford.csv'
df = pd.read_csv(oxfordprofil_path,sep=',')
matrix = [[] for x in xrange(len(df))]
#matrix = [[]*7916*7 for x in xrange(10)]

underLipBottom_x = {}
underLipBottom_y = {}
facialHair_mustache = {}
facialHair_beard = {}
facialHair_sideburns = {}
faceRectangle_top = {}

ip=0
index_arr=[]
#Index([u'underLipBottom_x', u'underLipBottom_y', u'facialHair_mustache'], dtype='object')
for index, row in df.iterrows():
    print ip
    userid = row['userId']
    underLipBottom_x = row['underLipBottom_x']
    underLipBottom_y = row['underLipBottom_y']
    facialHair_mustache = row['facialHair_mustache']
    faceRectangle_top = row['faceRectangle_top']
    facialHair_beard = row['facialHair_beard']
    facialHair_sideburns = row['facialHair_sideburns']
    matrix[index].append(userid)
    matrix[index].append(str(underLipBottom_x)) 
    matrix[index].append(str(underLipBottom_y))
    matrix[index].append(str(facialHair_mustache))
    matrix[index].append(str(faceRectangle_top))
    matrix[index].append(str(facialHair_beard))
    matrix[index].append(str(facialHair_sideburns))
    ip = ip+1
    index_arr.append(index)

faceDF = pd.DataFrame(matrix)
faceDF.columns=['userid', 'ulBx','ulBy','fhmu','ft','fhbeard','fhsideburns']
#gender image with face information, it is a DataFrame

age_image = pd.merge(ageDF,faceDF,on="userid")
facearr = age_image.as_matrix(columns=['ulBx','ulBy','fhmu','ft','fhbeard','fhsideburns'])
agenp= age_image.as_matrix(columns=['agegroup'])
X=facearr
n_features = facearr.shape[1]
print n_features #6
y = agenp.ravel()
print len(y) #7915

########################################################################
#fit model onevsrest
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
X_train, X_test, Y_train, Y_test = train_test_split(X, y,test_size=0.3,random_state=0)
classif = OneVsRestClassifier(SVC(kernel='rbf'))
clf = classif.fit(X_train, Y_train)
n_samples = X_train.shape[0]
scores = cross_validation.cross_val_score(clf, X_train, Y_train, cv=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

###############################################################################
#SVD
from sklearn import linear_model
from sklearn.decomposition import TruncatedSVD
from sklearn import linear_model
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import numpy as np
from sklearn import datasets

n_components = 6 
pca = TruncatedSVD(n_components=n_components)
# data = pca.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, y,test_size=0.3,random_state=0)
###after several experiments linear svc gave the best result:
clf = OneVsRestClassifier(LinearSVC(random_state=0))
clf2 = clf.fit(X_train, Y_train)
#cv = cross_validation.ShuffleSplit(n_samples, n_iter=5,test_size = 0.3, random_state = 0)
scores=cross_validation.cross_val_score(clf2, X_train, Y_train, cv=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# 54% array([ 0.47755544,  0.59859155,  0.53821138]) cv=3
# 0.42 (+/- 0.42) array([ 0.59009009,  0.59062218,  0.58972047,  0.09584087,  0.25497288]) cv=5
###############################################################################
#unsupervised SVM
from sklearn import svm
clf = svm.SVC()
data = X.as_matrix(columns=X.columns[1:])
label = age_image["agegroup"].tolist()
X_train, X_test, Y_train, Y_test = train_test_split(data,label,test_size=0.3,random_state=0)
clf2 = clf.fit(X_train, Y_train)
scores=cross_validation.cross_val_score(clf2, X_train, Y_train, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#non select key best, for all features svm 0.59 test_size=0.25, random_state=42 array([ 0.59046283,  0.59046283,  0.59046283,  0.59058989,  0.59100492])
###############################################################################
#knn
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, Y_train, Y_test = train_test_split(X, y,test_size=0.3,random_state=42)
neigh = KNeighborsClassifier(n_neighbors=90) #84
clf = neigh.fit(X_train, Y_train)
n_samples = X_train.shape[0]
scores = cross_validation.cross_val_score(clf, X_train, Y_train, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#0.60% array([ 0.59783589,  0.59783589,  0.59783589,  0.59837545,  0.59819005])
###############################################################################
#Bernoulli Naive Bayes
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
X_train, X_test, Y_train, Y_test = train_test_split(X, y,test_size=0.3,random_state=42)
clf2 = clf.fit(X_train, Y_train)
n_samples = X_train.shape[0]
scores = cross_validation.cross_val_score(clf2, X_train, Y_train, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#gender 58% array([ 0.57894737,  0.57894737,  0.57894737,  0.57894737,  0.57894737])
#age matrix 0.60 array([ 0.59783589,  0.59783589,  0.59693417,  0.59747292,  0.59819005])
