
faceCascade = cv2.CascadeClassifier('/home/ituser/Downloads/haarcascade_frontalface_alt.xmls/haarcascade_frontalface_alt.xml')
print faceCascade
gender_dic = {}
useri_arr = [] 
reader = csv.reader(open("/data/training/profile/profile.csv"))
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
    image = Image.open("/data/training/image/"+userid+".jpg").convert('L')   
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
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0)) #done in 68.299s
#h = 30 
#w = 30
#eigenfaces = pca.components_.reshape((n_components, h, w))
t0 = time()
#X_train_pca = X_train
X_train_pca = pca.transform(X_train)
print("done in %0.3fs" % (time() - t0)) #done in 6.222s
# ###############################################################################
# # Train a SVM classification model
print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
               'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0)) #done in 106.161s
print("Best estimator found by grid search:") #Best estimator found by grid search:
print(clf.best_estimator_) 

########################################################################
#Test Image Process

testuseri_arr = [] 
nofaceusrid_arr = []
for userid in nofaceusrid_arr:
     testuseri_arr.append(userid)

ip=0
nofaceimg_arr = []
value2 = []
for key in range(0,len(testuseri_arr)):
    userid = testuseri_arr[key]
    print ip
    image = Image.open(testpath+"image/"+userid+".jpg").convert('L')    
    image = np.array(image)
    image1 = cv2.resize(image,(30,30))
    image2 = image1.flatten()
    #print image2
    nofaceimg_arr.append(image2)
    ip = ip+1

########################################################################
# predict
X_test = np.array(nofaceimg_arr)
X_test_pca = pca.transform(X_test)
print("Predicting people's gender on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))   

d = {"userid":testuseri_arr, "gender": y_pred.tolist()} 




