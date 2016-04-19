#!/usr/local/bin/python
import sys
import csv
import cv2
from time import time
import random
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
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
from sklearn.datasets import samples_generator
from sklearn import svm

#argv[2]: /data/public-test-data/
#argv[4]: /output/
#testpath = '/Users/yaqunyu/UW_2016_Winter/TCSS555dataset/Public_Test/'

# try:
#     testpath = sys.argv[2]
#     outputpath = sys.argv[4]
#     print "Test Data is at " + testpath
#     print "Output Folder is: " + outputpath
# except IndexError as e:
#     print "ERROR: input paths are required: 1: path to image files 2: path to test data 3: output path"
#     sys.exit()

faceRectangle_width = {}
faceRectangle_height = {}
faceRectangle_left = {}
faceRectangle_top = {}
facialHair_beard = {}
facialHair_sideburns = {}

oxfordprofil_path = '/Users/yaqunyu/Desktop/oxford.csv'
df = pd.read_csv(oxfordprofil_path,sep=',')
df2 = pd.DataFrame
tempdf = pd.DataFrame
gendermatrix = [[] for x in xrange(9500)]
gender_dic = {}
useri_arr = [] 
genderdf = pd.read_csv('/Users/yaqunyu/UW_2016_Winter/TCSS555dataset/Train/Profile/Profile.csv',sep=',')
for index, row in genderdf.iterrows():
    userid = row['userid']
    gender = row['gender']
    gendermatrix[index].append(userid)
    gendermatrix[index].append(str(gender))

# for userid, faceRectangle_width,faceRectangle_height,faceRectangle_left, faceRectangle_top,facialHair_sideburns,facialHair_beard in reader:
#     useri_arr.append(userid)
#     faceRectangle_width[userid]=faceRectangle_width
#     faceRectangle_height[userid]=faceRectangle_height
#     faceRectangle_left[userid]=faceRectangle_left
#     faceRectangle_top[userid]=faceRectangle_top
#     facialHair_beard[userid]=facialHair_beard    
#     facialHair_sideburns[userid]=facialHair_sideburns

matrix = [[]*7916*7 for x in xrange(len(df))]
#matrix = [[]*7916*7 for x in xrange(10)]
ip=0
index_arr=[]
for index, row in df.iterrows():
    print ip
    userid = row['userId']
    faceRectangle_width = row['faceRectangle_width']
    faceRectangle_height = row['faceRectangle_height']
    faceRectangle_left = row['faceRectangle_left']
    faceRectangle_top = row['faceRectangle_top']
    facialHair_beard = row['facialHair_beard']
    facialHair_sideburns = row['facialHair_sideburns']
    matrix[index].append(userid)
    matrix[index].append(str(faceRectangle_width)) 
    matrix[index].append(str(faceRectangle_height))
    matrix[index].append(str(faceRectangle_left))
    matrix[index].append(str(faceRectangle_top))
    matrix[index].append(str(facialHair_beard))
    matrix[index].append(str(facialHair_sideburns))
    ip = ip+1
    index_arr.append(index)

genderDF = pd.DataFrame(gendermatrix)
genderDF.columns=['userid','gender']
faceDF = pd.DataFrame(matrix)
faceDF.columns=['userid', 'fw','fh','fl','ft','fhbeard','fhsideburns']
#gender image with face information, it is a DataFrame
gender_image = pd.merge(genderDF,faceDF,on="userid")
########################################################################
#check if userid coloum is in dataframe gender_image
if 'userid' in gender_image.columns:
    1
else:
    0
########################################################################
#create face infor numpy array facearr
facearr = gender_image.as_matrix(columns=['fw','fh','fl','ft','fhbeard','fhsideburns'])
########################################################################
#create gender infor numpy array gendenp
gendernp= gender_image.as_matrix(columns=['gender'])

########################################################################
#Training Image Process
X=facearr
n_features = facearr.shape[1]
print n_features
y = gendernp.ravel()
target_names = y
print len(y)
n_classes = target_names.shape[0] #19
print("Total dataset size:")
print("n_features: %d" % n_features) #900
print("n_classes: %d" % n_classes) #n_classes: 19
X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.25, random_state=42)
n_components = 200
print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0)) #done in 68.299s
#h = 30 
#w = 30
#eigenfaces = pca.components_.reshape((n_components, h, w))
t0 = time()
#X_train_pca = X_train

X_train_pca = pca.transform(X_train)
print("done in %0.3fs" % (time() - t0)) #done in 6.222s
###############################################################################
# Train a SVM classification model
print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
               'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0)) #done in 106.161s
print("Best estimator found by grid search:") #Best estimator found by grid search:
print(clf.best_estimator_) 

########################################################################
#Test Image Process for a class sklearn.pipeline.Pipeline(steps)
X, y = samples_generator.make_classification(n_informative=5, n_redundant=0, random_state=42)
anova_filter = SelectKBest(f_regression, k=5)
clf = svm.SVC(kernel='linear')
anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])
anova_svm.set_params(anova__k=10, svc__C=.1).fit(X, y)
prediction = anova_svm.predict(X)
anova_svm.score(X, y)

########################################################################
#Test Image Process
faceRectangle_width = {}
faceRectangle_height = {}
faceRectangle_left = {}
faceRectangle_top = {}
facialHair_beard = {}
facialHair_sideburns = {}

oxfordtest_path = '/Users/yaqunyu/Desktop/test/oxford.csv'
df = pd.read_csv(oxfordtest_path,sep=',')
df2 = pd.DataFrame
tempdf = pd.DataFrame
gendertestmatrix = [[] for x in xrange(9500)]
gender_dic = {}
useri_arr = [] 
genderdf = pd.read_csv('/data/test/profile/profile.csv',sep=',')
for index, row in genderdf.iterrows():
    userid = row['userid']
    gendertestmatrix[index].append(userid)

# for userid, faceRectangle_width,faceRectangle_height,faceRectangle_left, faceRectangle_top,facialHair_sideburns,facialHair_beard in reader:
#     useri_arr.append(userid)
#     faceRectangle_width[userid]=faceRectangle_width
#     faceRectangle_height[userid]=faceRectangle_height
#     faceRectangle_left[userid]=faceRectangle_left
#     faceRectangle_top[userid]=faceRectangle_top
#     facialHair_beard[userid]=facialHair_beard    
#     facialHair_sideburns[userid]=facialHair_sideburns

matrix = [[]*7916*7 for x in xrange(len(df))]
#matrix = [[]*7916*7 for x in xrange(10)]
ip=0
index_arr=[]
for index, row in df.iterrows():
    print ip
    userid = row['userId']
    faceRectangle_width = row['faceRectangle_width']
    faceRectangle_height = row['faceRectangle_height']
    faceRectangle_left = row['faceRectangle_left']
    faceRectangle_top = row['faceRectangle_top']
    facialHair_beard = row['facialHair_beard']
    facialHair_sideburns = row['facialHair_sideburns']
    matrix[index].append(userid)
    matrix[index].append(str(faceRectangle_width)) 
    matrix[index].append(str(faceRectangle_height))
    matrix[index].append(str(faceRectangle_left))
    matrix[index].append(str(faceRectangle_top))
    matrix[index].append(str(facialHair_beard))
    matrix[index].append(str(facialHair_sideburns))
    ip = ip+1
    index_arr.append(index)

faceDF = pd.DataFrame(matrix)
faceDF.columns=['userid', 'fw','fh','fl','ft','fhbeard','fhsideburns']
#gender image with face information, it is a DataFrame

########################################################################
#check if userid coloum is in dataframe gender_image
if 'userid' in faceDF.columns:
    1
else:
    0
########################################################################
#create face infor numpy array facearr

facearr = faceDF.as_matrix(columns=['fw','fh','fl','ft','fhbeard','fhsideburns'])

########################################################################
#Predict
X_test=facearr
n_features = facearr.shape[1]
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
# #outputpath ="/Users/yaqunyu/Desktop/image_python/TCSS555TestOutput/"
def toXML(row,filename,mode="w"):
     xml = ['<']
     #for field in row.index:
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
     dir = os.path.dirname(filename)
     #if not os.path.exists(dir):
      #   os.makedirs(dir)
     with open(filename,mode) as f:
         f.write(res)

df.apply(lambda row: toXML(row, outputpath + row['userid'] + ".xml"), axis=1)     
# for user_id in testuseri_arr:
#  	filename=user_id
#  	res = d.apply(toXML, axis=1)
#  	# if filename is None:
# 		# return res
#  	with open(path+filename+'.xml', mode='w') as f:
#  		f.write(res)

print "Successful write to files"
###############################################################################
#output for each userID
# for user_id in testuseri_arr:
# 	pd.DataFrame.to_xml = outputXML     
# 	d.outputXML(user_id+'.xml')


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

#####crossed Validation######################################################
n_samples = X_train_pca.shape[0]
# cv = cross_validation.ShuffleSplit(n_samples, n_iter=5,test_size = 0.3, random_state = 0)
scores=cross_validation.cross_val_score(clf, X_train_pca, y_train, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#range(0,5501), n_components=200, 30x30, 0.25, 0.68 random_state=42
#range(0,5501), n_components=300, 100x100, 0.25 0.66 random_state=5
#range(0,5501), n_components=300, 100x100, 0.3 0.65778316172 random_state=5
#range(0,6501), n_components=300, 30x30, 0.3 0.678113787801 random_state=5
#range(0,6501), n_components=300, 30x30, 0.25 0.686346863469 random_state=5
#range(0,9500), n_components=200, 30x30, 0.25 0.709175084175 random_state=5
#range(0,9500), n_components=600, 30x30, 0.25 0.645622895623 random_state=5
#range(0,9500), n_components=200, 30x30, 0.25 0.712121212121 random_state=42





##age 










