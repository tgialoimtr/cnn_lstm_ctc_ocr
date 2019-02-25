'''
Created on Nov 3, 2018

@author: loitg
'''
import os, sys
import numpy as np
from sklearn.externals import joblib

def readData(path):
    datafile = open(path,'r')
    data = []
    def _toFloat(a):
        if a == 'True':
            return 1
        elif a == 'False':
            return 0
        elif a == 'nan':
            return np.nan
        elif len(a) >=3 and a[-3:] == 'inf':
            return 99
        else:
            return float(a)
        
    for line in datafile.readlines():
        fields = line.rstrip().split(',')
        fields = [_toFloat(x) for x in fields]
        data.append(fields)
        
    return np.array(data)
    
def buildModel1(data, clfname, modelpath):
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    
    clf_names = ["SVM",
                "NN",
                 "DecitionTree"]
    clfs = [SVC(gamma=2, C=1, probability=True),
            MLPClassifier(hidden_layer_sizes=(30,10,5), alpha=1),
           DecisionTreeClassifier(max_depth=4)]
    
    X = data[:, :13]
    print data.shape
    y = (data[:, 14] >0.5) | (data[:, 13] > 0.5)
    for clf, name in zip(clfs, clf_names):
        print "===---==" + name
        for rs in range(30,33):
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=rs)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            print "--=====" + str(score)
            
    for clf, name in zip(clfs, clf_names):
        if name == clfname:
            clf.fit(X,y)
            joblib.dump(clf, modelpath)


if __name__ == "__main__":
#     sys.argv = [None, None]
#     sys.argv[0] = '/home/ntnnhi/workspace/eclipse/ocr-app/resources/le_model/le_cls_data_3.csv'
#     sys.argv[1] = '/home/ntnnhi/workspace/eclipse/ocr-app/resources/le_model/le_model_4.pkl'
    data = readData(sys.argv[1])
    data = data[~np.isnan(data).any(axis=1)]
    buildModel1(data,"SVM", sys.argv[2])
     
     
    sys.exit(0)

# if __name__ == "__main__":
#     gtdatafile = open(le_cls_data_path, 'w')
#     for filename in os.listdir('/home/loitg/Downloads/complex-bg/special_line/'):
#         if filename[-3:].upper() == 'JPG':
# #         if filename == "4.JPG":
#             print filename
# #             print dir(random)
# #             sys.exit(0)
#             buildLineRecogData('/home/loitg/Downloads/complex-bg/special_line/' + filename, gtdatafile)
            
            
            