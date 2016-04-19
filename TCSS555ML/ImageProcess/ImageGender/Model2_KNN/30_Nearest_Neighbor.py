import sys
import csv
import cv2
import math
import random

import numpy as np
from sklearn import cross_validation
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction import image
import os
from PIL import Image
faceCascade = cv2.CascadeClassifier('/Users/yaqunyu/anaconda/lib/python2.7/site-packages/opencv/data/haarcascades/haarcascade_frontalface_alt.xml')
# image process
trainingdata_path_root = '/Users/yaqunyu/UW_2016_Winter/TCSS555dataset/'
trainingimage_path_root = trainingdata_path_root + 'Train/Image/'
traininglabels_path = trainingdata_path_root + 'Train/Profile/Profile.csv'
#read in profile.csv
# trainingdata_path_root = '/Users/yaqunyu/Desktop/'
# trainingimage_path_root =trainingdata_path_root +'image_python/train/'
# traininglabels_path = '/Users/yaqunyu/UW_2016_Winter/TCSS555dataset/Train/Profile/Profile.csv'
gender_dic = {}
useri_arr = [] 
true = 0
false = 0
reader = csv.reader(open("/Users/yaqunyu/UW_2016_Winter/TCSS555dataset/Train/Profile/Profile.csv"))
#reader = csv.reader(open("/Users/yaqunyu/Desktop/test.csv",'rU'))
#for number, userid , gender in reader:
for number , userid , age ,gender,ope , con , ext , agr , neu in reader:
    useri_arr.append(userid)
    gender_dic[userid]=gender

print len(useri_arr)
print gender_dic
neigh = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=30, p=2,
           weights='uniform')


useri_arr = useri_arr[1:]
ip=0
output = []
value1 = []
for key in range(0,7001):
    userid = useri_arr[key]
    value = gender_dic[userid]
    print ip
    g = int(float(value)) 
    value1.append(g)
    #value1.append(g)
    #print value1
    image = Image.open("/Users/yaqunyu/UW_2016_Winter/TCSS555dataset/Train/Image/"+userid+".jpg").convert('L')    
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

output=np.array(output)
neigh.fit(output,value1)
ip = 7001

for key in range(7001,9500):
    userid = useri_arr[key]
    value = gender_dic[userid]
    print ip
    g = int(float(value)) 
    value1.append(g)
    #print value1
    image = Image.open("/Users/yaqunyu/UW_2016_Winter/TCSS555dataset/Train/Image/"+userid+".jpg").convert('L')
    #image = Image.open("/Users/yaqunyu/Desktop/image_python/training/"+userid+".jpg").convert('L')
    image = np.array(image)
    #print el
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(image, 1.1, 5) #when there is only one image, 5 will change to 2 or 0
    for (x, y, w, h) in faces:
        crop_image = image[y:y+h, x:x+w]
        resize_face = cv2.resize(crop_image,(30,30))
        X= np.array(resize_face)
        arr=X.tolist()
        #print arr
        va = []
        for i in range(0,len(arr)):
            va.extend(arr[i])   
    output.append(va)
    #print output              
    ip=ip+1

output=np.array(output)
r=neigh.predict(output)
#print r
#print len(r)
#print len(value1)
for j in range(0,len(r)):
    if (r[j] == value1[j]):
        true = true +1
    else:
        false = false +1 

print "--------finish testing--------"
print "true : " ,true
print "false : " , false
total = true + false
print total
accuracy = float(true) / float(total)
print "accuracy: ", accuracy



#-------------------Cross Validation
output = np.array(output)
n_samples = output.shape[0]
neigh = KNeighborsClassifier(n_neighbors=90)
clf = neigh.fit(output,value1)
cv = cross_validation.ShuffleSplit(n_samples, n_iter=5,test_size = 0.3, random_state = 42)
scores=cross_validation.cross_val_score(clf, output, value1, cv=cv)
scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

array([ 0.61438596,  0.64491228,  0.64140351,  0.64982456,  0.64210526])
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
Accuracy: 0.64 (+/- 0.02)


# K
#     n_samples = 9500 n_iter=5,test_size = 0.3 random_state = 42                     
#     Accuracy1   Accuracy2   Accuracy3   Accuracy4   Accuracy5   Accuracy    
# 5                           
# 30                          
# 90                          
#     n_samples = 9500 n_iter=5,test_size = 0.25 random_state = 42                        
#     Accuracy1   Accuracy2   Accuracy3   Accuracy4   Accuracy5   Accuracy    
# 5   0.63747368  0.61852632  0.60968421  0.61726316  0.62273684  0.62 (+/- 0.02) 
# 30  0.66652632  0.64968421  0.63957895  0.656   0.66568421  0.66 (+/- 0.02) 
# 90  0.6 0.62666667  0.59111111  0.6 0.6 0.60 (+/- 0.02) 
#     n_samples = 9500 n_iter=5,test_size = 0.4 random_state = 42                     
#     Accuracy1   Accuracy2   Accuracy3   Accuracy4   Accuracy5   Accuracy    
# 5   0.63342105  0.62236842  0.62    0.62210526  0.60684211   0.62 (+/- 0.02)    
# 30  0.65052632  0.64394737  0.64815789  0.64578947  0.65315789  0.65 (+/- 0.01) 
# 90  0.64210526  0.64210526  0.63894737  0.64289474  0.64947368  0.64 (+/- 0.01) 






