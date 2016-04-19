'''
Created on Mar 2, 2016

@author: lyulan
'''

from os import listdir
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors.regression import KNeighborsRegressor
import io
import re
import nltk
import csv
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest
# override the f_regression task used in SelectKBest 
def f_regression(X, Y):
    import sklearn.feature_selection
    return sklearn.feature_selection.f_regression(X, Y, center=False)

# ===================define tokenizor======================
# =========================================================
# load nltk's SnowballStemmer as variabled 'stemmer'
stemmer = SnowballStemmer("english")

# define a tokenizer and stemmer which returns the set of stems in the text that it is passed
def  tokenize(text):
    filtered_tokens = []
    # filter ":)", ":(", ":/", ":|", ":D", ":O" and ":-)" emotion
    pattern_emotion1 = re.compile("[:][)]|[:][(]|[:][/]|[:][|]|[:][D]|[:][O]|[:][-][)]")
    # filter "^_^", "-_-", "><", ">.<", "o_o" and "<3" emotion
    pattern_emotion2 = re.compile("[\^][_][\^]|[-][_][-]|[>][<]|[>][.][<]|[o][_][o]|[<][3]") 
    emotions1 = list(set(re.findall(pattern_emotion1, text)))
    emotions2 = list(set(re.findall(pattern_emotion2, text)))
    
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]    
    # filter out any tokens not containing letters (e.g., numeric tokens, punctuation tokens)
    letters = []
    for token in tokens:
        letters.extend(re.findall("[a-zA-Z]+", token))
    stems = []
    for item in letters:
        raw_stem = stemmer.stem(item)
        # remove single-character stem
        if len(raw_stem) > 1:
            stems.append(raw_stem)
    stems = list(set(stems))    
        
    filtered_tokens.extend(emotions1)
    filtered_tokens.extend(emotions2)
    filtered_tokens.extend(stems)  
    return filtered_tokens


def predict_personality(test_path):
    #train_path = '/Users/lyulan/TCSS555/Data/Train'
    train_path = '/data/train/'
    print 'train folder is: ', train_path
    
    # prepare training data
    personality_dic = {} 
    profile_path = train_path + '/Profile/Profile.csv'
    with open(profile_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            userid = row['userid']
            personality = []
            personality.append(float(row['ope']))
            personality.append(float(row['con']))
            personality.append(float(row['ext']))
            personality.append(float(row['agr']))
            personality.append(float(row['neu']))
            personality_dic[userid] = personality
    #print personality_dic

    train_userids = []
    train_texts = []
    train_personalities = []
    #train_filepath = train_path + '/Text'
    train_filepath = train_path + '/text'
    train_files = [f for f in listdir(train_filepath) if f.endswith(".txt")]
    for n in train_files:
        f = io.open(train_filepath + '/' + n, 'r', encoding='latin-1')
        text = f.read()
        userid = n.replace('.txt', '')
        f.close()
        # get rid of users without personality data
        if userid in personality_dic:
            train_userids.append(userid)
            train_texts.append(text)
            train_personalities.append(personality_dic[userid])
    
    # prepare testing data
    test_userids = []
    test_texts = []
    #test_filepath = test_path + '/Text'
    test_filepath = test_path + '/text'
    test_files = [f for f in listdir(test_filepath) if f.endswith(".txt")]
    for n in test_files:
        f = io.open(test_filepath + '/' + n, 'r', encoding='latin-1')
        text = f.read()
        userid = n.replace('.txt', '')
        f.close()
        test_userids.append(userid)
        test_texts.append(text)
        
        
    # extract features from training data using tfidf_vectorizer
    tfidf_vectorizer = TfidfVectorizer(min_df=0.005, max_df=0.3, ngram_range=(1,3), 
                                       stop_words='english',use_idf=True, tokenizer=tokenize)
    trainX = tfidf_vectorizer.fit_transform(train_texts)
    print("Training data: n_samples: %d, n_features: %d" % trainX.shape)
    # extract features from testing data using the same tfidf_vectorizer
    testX = tfidf_vectorizer.transform(test_texts)
    print("Testing data: n_samples: %d, n_features: %d" % testX.shape)
    # get feature names extracted by tfidf_vectorizer
    feature_names = tfidf_vectorizer.get_feature_names()
    print(feature_names)


    result = {}
    result['userid'] = test_userids
    
    for x in range(0, 5):
        y_train = np.array(train_personalities)[:,x]
        
        # reduce features by SelectKBest
        nkb_features = 50
        skb = SelectKBest(score_func=f_regression, k=nkb_features)
        X_train = skb.fit_transform(trainX, y_train)
        X_test = skb.transform(testX)
        
        # train using KNNRegressor
        knn = KNeighborsRegressor(n_neighbors=500)
        knn.fit(X_train, y_train)
        predicted = knn.predict(X_test)
        
        if x==0:
            result['ope'] = predicted
        elif x==1:
            result['con'] = predicted
        elif x==2:
            result['ext'] = predicted
        elif x==3:
            result['agr'] = predicted
        else:
            result['neu'] = predicted
    
    result_df = pd.DataFrame(result, columns=['userid', 'ope', 'con', 'ext', 'agr', 'neu'])
    return result_df
    

