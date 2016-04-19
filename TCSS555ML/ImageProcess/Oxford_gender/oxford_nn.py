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
from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os
from PIL import Image
import pandas as pd
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

# faceRectangle_width = {}
# faceRectangle_height = {}
# faceRectangle_left = {}
# faceRectangle_top = {}
# facialHair_beard = {}
# facialHair_sideburns = {}

# oxfordprofil_path = '/Users/yaqunyu/Desktop/oxford.csv'
# df = pd.read_csv(oxfordprofil_path,sep=',')
# df2 = pd.DataFrame
# tempdf = pd.DataFrame
# gendermatrix = [[] for x in xrange(9500)]
# gender_dic = {}
# useri_arr = [] 
# genderdf = pd.read_csv('/Users/yaqunyu/UW_2016_Winter/TCSS555dataset/Train/Profile/Profile.csv',sep=',')
# for index, row in genderdf.iterrows():
#     userid = row['userid']
#     gender = row['gender']
#     gendermatrix[index].append(userid)
#     gendermatrix[index].append(str(gender))

# # for userid, faceRectangle_width,faceRectangle_height,faceRectangle_left, faceRectangle_top,facialHair_sideburns,facialHair_beard in reader:
# #     useri_arr.append(userid)
# #     faceRectangle_width[userid]=faceRectangle_width
# #     faceRectangle_height[userid]=faceRectangle_height
# #     faceRectangle_left[userid]=faceRectangle_left
# #     faceRectangle_top[userid]=faceRectangle_top
# #     facialHair_beard[userid]=facialHair_beard    
# #     facialHair_sideburns[userid]=facialHair_sideburns

# matrix = [[]*7916*7 for x in xrange(len(df))]
# #matrix = [[]*7916*7 for x in xrange(10)]
# ip=0
# index_arr=[]
# for index, row in df.iterrows():
#     print ip
#     userid = row['userId']
#     faceRectangle_width = row['faceRectangle_width']
#     faceRectangle_height = row['faceRectangle_height']
#     faceRectangle_left = row['faceRectangle_left']
#     faceRectangle_top = row['faceRectangle_top']
#     facialHair_beard = row['facialHair_beard']
#     facialHair_sideburns = row['facialHair_sideburns']
#     matrix[index].append(userid)
#     matrix[index].append(str(faceRectangle_width)) 
#     matrix[index].append(str(faceRectangle_height))
#     matrix[index].append(str(faceRectangle_left))
#     matrix[index].append(str(faceRectangle_top))
#     matrix[index].append(str(facialHair_beard))
#     matrix[index].append(str(facialHair_sideburns))
#     ip = ip+1
#     index_arr.append(index)

# genderDF = pd.DataFrame(gendermatrix)
# genderDF.columns=['userid','gender']
# faceDF = pd.DataFrame(matrix)
# faceDF.columns=['userid', 'fw','fh','fl','ft','fhbeard','fhsideburns']
# #gender image with face information, it is a DataFrame
# gender_image = pd.merge(genderDF,faceDF,on="userid")





underLipBottom_x = {}
underLipBottom_y = {}
facialHair_mustache = {}
facialHair_beard = {}
facialHair_sideburns = {}
faceRectangle_top = {}

oxfordprofil_path = '/Users/yaqunyu/UW_2016_Winter/oxford.csv'
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

genderDF = pd.DataFrame(gendermatrix)
genderDF.columns=['userid','gender']
faceDF = pd.DataFrame(matrix)
faceDF.columns=['userid', 'ulBx','ulBy','fhmu','ft','fhbeard','fhsideburns']
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

facearr = gender_image.as_matrix(columns=['ulBx','ulBy','fhmu','ft','fhbeard','fhsideburns'])

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
X_train, X_test, Y_train, Y_test = train_test_split(X, y,test_size=0.3,random_state=0)
logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)
classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

###############################################################################
# Training Start

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
# cv = cross_validation.ShuffleSplit(n_samples, n_iter=5,test_size = 0.3, random_state = 0)
scores = cross_validation.cross_val_score(clf2, X_train, Y_train, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# 81%

# ########################################################################
# #Test Image Process
testuseri_arr = [] 
reader = csv.reader(open(testpath + 'profile/profile.csv'))
for number , userid , age ,gender,ope , con , ext , agr , neu in reader:
     testuseri_arr.append(userid)

testuseri_arr = testuseri_arr[1:]
ip=0
output2 = []
value2 = []
for key in range(0,len(testuseri_arr)):
    userid = testuseri_arr[key]
    #value = gender_dic[userid]
    print ip
    #value1.append(g)
    #print value1
    image = Image.open(testpath+"image/"+userid+".jpg").convert('L')    
    image = np.array(image)
    #faces = faceCascade.detectMultiScale(image, 1.1, 5) #when there is only one image, 5 will change to 2 or 0
    image1 = cv2.resize(image,(30,30))
    image2 = image1.flatten()
    #print image2
    output2.append(image2)
    ip = ip+1
     
########################################################################
#Predict
X_test = np.array(output2)
X_test_pca = X_test
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

# #####crossed Validation######################################################
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










