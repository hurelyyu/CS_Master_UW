#!/usr/bin/env python
import sys
import csv
import cv2
from time import time
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA
from sklearn.feature_extraction import image
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os
from PIL import Image
import pandas as pd
from lxml import objectify


#testpath = '/Users/yaqunyu/UW_2016_Winter/TCSS555dataset/Public_Test/'
try:
    testpath = sys.argv[1]
    outputpath = sys.argv[2]
    print "Test Data is at " + testpath
    print "Output Folder is: " + outputpath
except IndexError as e:
    print "ERROR: input paths are required: 1: path to image files 2: path to test data 3: output path"
    sys.exit()

faceCascade = cv2.CascadeClassifier('/Users/yaqunyu/anaconda/lib/python2.7/site-packages/opencv/data/haarcascades/haarcascade_frontalface_alt.xml')

gender_dic = {}
useri_arr = [] 
reader = csv.reader(open("/data/training/profile.csv"))
for number , userid , age ,gender,ope , con , ext , agr , neu in reader:
    useri_arr.append(userid)
    gender_dic[userid]=gender
########################################################################
#Training Image Process

ip=0
output = []
value1 = []
for key in range(1,len(useri_arr)):
    userid = useri_arr[key]
    value = gender_dic[userid]
    print ip
    g = int(float(value)) 
    value1.append(g)
    #value1.append(g)
    #print value1
    image = Image.open("/data/training/image/"+userid+".jpg").convert('L')    
    #image = Image.open("/Users/yaqunyu/Desktop/image_python/training/"+userid+".jpg").convert('L')
    image = np.array(image)
    #print el
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(image, 1.1, 5) #when there is only one image, 5 will change to 2 or 0
    for (x, y, w, h) in faces:
        crop_image = image[y:y+h, x:x+w]
        resize_face = cv2.resize(crop_image,(30,30))
        #X= np.array(resize_face)
        arr=resize_face.tolist()
        #print arr
        va = []
        for i in range(0,len(arr)):
            va.extend(arr[i])
        #print va 
    output.append(va)
    #print output              
    ip=ip+1

X = np.array(output)
n_features = X.shape[1]
y = np.array(value1)
target_names = y
n_classes = target_names.shape[0] #19
print("Total dataset size:")
print("n_features: %d" % n_features) #900
print("n_classes: %d" % n_classes) #n_classes: 19
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0, random_state=42)
n_components = 200
print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))

#Extracting the top 200 eigenfaces from 13 faces
t0 = time()
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0)) #done in 68.299s
h = 30 
w = 30
eigenfaces = pca.components_.reshape((n_components, h, w))
t0 = time()
X_train_pca = pca.transform(X_train)
print("done in %0.3fs" % (time() - t0)) #done in 6.222s
###############################################################################
# Train a SVM classification model
print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0)) #done in 106.161s
print("Best estimator found by grid search:") #Best estimator found by grid search:
print(clf.best_estimator_) 

########################################################################
#Test Image Process
testuseri_arr = [] 
reader = csv.reader(open(testpath+'Profile/Profile.csv'))
for number , userid , age ,gender,ope , con , ext , agr , neu in reader:
    testuseri_arr.append(userid)

testuseri_arr = testuseri_arr[1:]
ip=0
output2 = []
value2 = []
for key in range(1,len(testuseri_arr)):
    userid = testuseri_arr[key]
    #value = gender_dic[userid]
    print ip
    #value1.append(g)
    #print value1
    image = Image.open(testpath+"Image/"+userid+".jpg").convert('L')    
    image = np.array(image)
    faces = faceCascade.detectMultiScale(image, 1.1, 5) #when there is only one image, 5 will change to 2 or 0
    for (x, y, w, h) in faces:
        crop_image = image[y:y+h, x:x+w]
        resize_face = cv2.resize(crop_image,(30,30))
        arr=resize_face.tolist()
        va2 = []
        for i in range(0,len(arr)):
            va2.extend(arr[i])
    output2.append(va2)           
    ip=ip+1

########################################################################
#Predict
X_test = np.array(output2)
X_test_pca = pca.transform(X_test)
print("Predicting people's gender on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

########################################################################
#Convert frame, to xml
#testuseri_arr = testuseri_arr[1:]
d = {"userid":testuseri_arr,"age_group":None, "gender": y_pred.tolist(),"extrovert":None, "neurotic":None, "agreeable":None, "conscientious":None, "open":None}
df = pd.DataFrame(d, columns=['userid', 'age_group', 'gender', 'extrovert', 'neurotic', 'agreeable', 'conscientious', 'open'])

#outputpath ="/Users/yaqunyu/Desktop/image_python/TCSS555TestOutput/"

def toXML(row,filename,mode="w"):
    xml = ['<']
    # for field in row.index:
    xml.append('  userId="{0}"'.format(row["userid"]))
    xml.append('  age_group="{0}"'.format(row["age_group"]))
    xml.append('  gender="{0}"'.format(row["gender"]))
    xml.append('  extrovert="{0}"'.format(row["extrovert"]))
    xml.append('  neurotic="{0}"'.format(row["neurotic"]))
    xml.append('  agreeable="{0}"'.format(row["agreeable"]))
    xml.append('  conscientious="{0}"'.format(row["conscientious"]))
    xml.append('  open="{0}"'.format(row["open"]))
    xml.append('/>')
    res = '\n'.join(xml)
    with open(filename,mode) as f:
        f.write(res)

df.apply(lambda row: toXML(row, outputpath + row['userid'] + ".xml"), axis=1)  
      

# for user_id in testuseri_arr:
# 	filename=user_id
# 	res = d.apply(toXML, axis=1)
# 	if filename is None:
# 		return res
# 	with open(path+filename+'.xml', mode='w') as f:
# 		f.write(res)

	
###############################################################################
#output for each userID
for user_id in testuseri_arr:
	pd.DataFrame.to_xml = outputXML     
	d.outputXML(user_id+'.xml')


#SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,
#  decision_function_shape=None, degree=3, gamma=0.0001, kernel='rbf',
#  max_iter=-1, probability=False, random_state=None, shrinking=True,
#  tol=0.001, verbose=False)
###############################################################################
# Quantitative evaluation of the model quality on the test set
# print("Predicting people's gender on the test set")
# true = 0
# false = 0
# t0 = time()
# y_pred = clf.predict(X_test_pca)
# print("done in %0.3fs" % (time() - t0))
# y_pred
# y_test
# for j in range(-1,len(y_test)):
#     if (y_pred[j] == y_test[j]):
#         true = true +1
#     else:
#         false = false +1

# print "true : " ,true 
# print "false : " , false 
# total = true + false 
# print total
# accuracy = float(true) / float(total)
# print "accuracy: ", accuracy

# #####crossed Validation######################################################
# n_samples = X_train_pca.shape[0]
# #cv = cross_validation.ShuffleSplit(n_samples, n_iter=5,test_size = 0.3, random_state = 0)
# scores=cross_validation.cross_val_score(clf, X_train_pca, y_train, cv=3)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#range(0,5501), n_components=200, 30x30, 0.25, 0.68 random_state=42
#range(0,5501), n_components=300, 100x100, 0.25 0.66 random_state=5
#range(0,5501), n_components=300, 100x100, 0.3 0.65778316172 random_state=5
#range(0,6501), n_components=300, 30x30, 0.3 0.678113787801 random_state=5
#range(0,6501), n_components=300, 30x30, 0.25 0.686346863469 random_state=5
#range(0,9500), n_components=200, 30x30, 0.25 0.709175084175 random_state=5
#range(0,9500), n_components=600, 30x30, 0.25 0.645622895623 random_state=5
#range(0,9500), n_components=200, 30x30, 0.25 0.712121212121 random_state=42
















