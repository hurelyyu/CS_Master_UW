This Python file contains Oxford predict age_group，it separte into two part:
1. Using Select K best for feature selection
2. Using Manully select features related close to age_group

In the first part we follow the order:
1. convert oxford information into pandas data frame
2. Input data frame and select the k=5 best features by:
    1> f_classif 
    2> Chi2: since Chi2 do not take negative data, we reconstruct our dataframe by data.clip(0) to convert each negative into 0
3. Using output from last step as our final data input for fit model, the model we consider here is:
    1> knn
    2> OneVsRestClassifier
    3> SVD
    4> SVM
    5> Bernoulli Naive Bayes

In the second part we follow the same order:    
1. convert oxford information into pandas data frame
2. Input data frame and manually select features:
    underLipBottom_x = {}
    underLipBottom_y = {}
    facialHair_mustache = {}
    facialHair_beard = {}
    facialHair_sideburns = {}
    faceRectangle_top = {}
3. Using output from last step as our final data input for fit model, the model we consider here is:
    1> knn
    2> OneVsRestClassifier
    3> SVD
    4> SVM
    5> Bernoulli Naive Bayes
