In AgeUseOxfordVM.py
We have the same function with our final version on VM
it splitted into four parts:
Part one: gender prediction
Part two: age prediction
part three: personality prediction
part four: combine all prediction result output in certain format as .xml

Part one: gender prediction
1> Oxford file image information predict use knn for user who have face
2> Normal image process using 9500 as traindata only predict gender for user who do not have face in Oxford by knn

Part two: age prediction
1> Oxford file image information predict use knn for user who have face
2> Pur image process using 9500 as traindata only predict age for user who do not have face in Oxford by knn
