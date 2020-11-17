from __future__ import print_function

import glob
import os
import pickle

import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from xlwt import Workbook
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

from feature_selection.wrapper import SFS_validate

def LR_evaluation_procedure(LR, X_train, Y_train, X_test, Y_test):
    print ('Starting with Logistic Regression procedure', '\n')

    LR.fit(X_train, Y_train)

    filename = 'saved_models/SFS_LR_Model.pkl'
    pickle.dump(LR, open(filename, 'wb'))

    predict = LR.predict(X_test)

    print ('The confusion matrix is')
    cm = confusion_matrix(Y_test, predict)
    print (cm, '\n')

    diagonal_sum = 0
    for i in cm.diagonal():
        diagonal_sum += i

    confusion_accuracy = ((diagonal_sum)/float(cm.sum())) * 100
    print ('The accuracy of Logistic Regression classifier is ', confusion_accuracy, "%", '\n')
    return confusion_accuracy

def SVM_evaluation_procedure(SVC, X_train, Y_train, X_test, Y_test):
    print ('Starting with the SVM procedure', '\n')

    SVC.fit(X_train, Y_train)

    filename = 'saved_models/SFS_SVM_Model.pkl'
    pickle.dump(SVC, open(filename, 'wb'))

    predict = SVC.predict(X_test)

    print ('The confusion matrix is')
    cm = confusion_matrix(Y_test, predict)
    print (cm, '\n')

    diagonal_sum = 0
    for i in cm.diagonal():
        diagonal_sum += i

    confusion_accuracy = ((diagonal_sum)/float(cm.sum())) * 100
    print ('The accuracy of SVM classifier is ', confusion_accuracy, "%", '\n')
    return confusion_accuracy

def KNN_evaluation_procedure(KNN, X_train, Y_train, X_test, Y_test):
    print ('Starting with the KNN procedure', '\n')

    # X_train = preprocessing.scale(X_train)
    # X_test = preprocessing.scale(X_test)

    KNN.fit(X_train, Y_train)

    filename = 'saved_models/SFS_KNN_Model.pkl'
    pickle.dump(KNN, open(filename, 'wb'))

    predict = KNN.predict(X_test)

    print ('The confusion matrix is')
    cm = confusion_matrix(Y_test, predict)
    print (cm, '\n')

    diagonal_sum = 0
    for i in cm.diagonal():
        diagonal_sum += i

    confusion_accuracy = ((diagonal_sum)/float(cm.sum())) * 100
    print ('The accuracy of KNN classifier is ', confusion_accuracy, "%", '\n')
    return confusion_accuracy


def RF_evaluation_procedure(RF, X_train, Y_train, X_test, Y_test):
    print ('Starting with Random Forest procedure', '\n')

    RF.fit(X_train, Y_train)

    filename = 'saved_models/SFS_RF_Model.pkl'
    pickle.dump(RF, open(filename, 'wb'))

    predict = RF.predict(X_test)

    print ('The confusion matrix is')
    cm = confusion_matrix(Y_test, predict)
    print (cm, '\n')

    diagonal_sum = 0
    for i in  cm.diagonal():
        diagonal_sum += i

    confusion_accuracy = ((diagonal_sum)/float(cm.sum())) * 100
    print ('The accuracy of Random Forest classifier is ', confusion_accuracy,"%", '\n')
    return confusion_accuracy

def read_features(base_dir_train, base_dir_test, sub_dir_train, sub_dir_test):
    D_train = []
    L_train = []

    for label, class_names in enumerate(sub_dir_train, start = 0):
        # print("label and class_names ",label, class_names)
        mvector_fft_path = os.path.join(base_dir_train, class_names, "pyaudio_features", "*.mvector.npy")
        all_files = glob.glob(mvector_fft_path)
        for f in all_files:
            value = np.load(f)
            D_train.append(value[:])
            L_train.append(label)
        # print(D_train)

    D_test = []
    L_test = []

    for label, class_names in enumerate(sub_dir_test, start = 0):
        mvector_fft_path = os.path.join(base_dir_test, class_names, "pyaudio_features", "*.mvector.npy")
        all_files = glob.glob(mvector_fft_path)
        for f in all_files:
            value = np.load(f)
            D_test.append(value[:])
            L_test.append(label)

    return np.array(D_train), np.array(L_train), np.array(D_test), np.array(L_test)


wb = Workbook()

rowNum = 1

def runFeatures_RF(i, numberOfFeature, X_train, Y_train, X_test, Y_test):
    print("Number of features Selected RF : " ,numberOfFeature)
    RF = RandomForestClassifier(n_estimators = 100, random_state = 1, max_features = 'log2')
    sfs1 = sfs(RF, k_features=numberOfFeature, forward=True, floating=False, verbose=2, scoring='accuracy', cv=0, n_jobs=-1)
    sfs1.fit(X_train, Y_train)
    cols = sfs1.k_feature_idx_
    print('The indices of best features RF are: ', cols, '\n')
    str1 = ','.join(str(e) for e in cols)

    X_train = sfs1.transform(X_train)
    X_test = sfs1.transform(X_test)

    filename = 'saved_models/SFS_RF.pkl'
    pickle.dump(sfs1, open(filename, 'wb'))
    train_acc = RF_evaluation_procedure(RF ,X_train, Y_train, X_test, Y_test)
    X_valid, Y_valid = SFS_validate.read_features(i)
    sfs1 = pickle.load(open(filename, 'rb'))
    X_valid = sfs1.transform(X_valid)
    val_acc = SFS_validate.validation_procedure_RF(X_valid, Y_valid)
    print("val acc runFeatures_RF",val_acc)
    return str1 ,train_acc, val_acc

def runFeatures_LR(i, numberOfFeature, X_train, Y_train, X_test, Y_test):
    print("Number of features Selected LR: ", numberOfFeature)
    LR = LogisticRegression(penalty = 'l1', tol = 0.1, random_state = 12)
    sfs1 = sfs(LR, k_features = numberOfFeature, forward = True, floating = False, verbose = 2, scoring = 'accuracy', cv = 0, n_jobs = -1)
    sfs1.fit(X_train, Y_train)
    cols = sfs1.k_feature_idx_
    print('The indices of best features LR are: ', cols, '\n')
    str1 = ','.join(str(e) for e in cols)

    X_train = sfs1.transform(X_train)
    X_test = sfs1.transform(X_test)

    filename = 'saved_models/SFS_LR.pkl'
    pickle.dump(sfs1, open(filename, 'wb'))
    train_acc = LR_evaluation_procedure(LR, X_train, Y_train, X_test, Y_test)
    X_valid, Y_valid = SFS_validate.read_features(i)
    sfs1 = pickle.load(open(filename, 'rb'))
    X_valid = sfs1.transform(X_valid)
    val_acc = SFS_validate.validation_procedure_LR(X_valid, Y_valid)
    return str1, train_acc, val_acc

def runFeatures_SVM(i, numberOfFeature, X_train, Y_train, X_test, Y_test):
    print("Number of features Selected SVM : ", numberOfFeature)
    SVC1 = SVC(kernel = 'linear', probability = True, random_state = 0)
    sfs1 = sfs(SVC1, k_features = numberOfFeature, forward = True, floating = False, verbose = 2, scoring = 'accuracy', cv = 0, n_jobs = -1)
    sfs1.fit(X_train, Y_train)
    cols = sfs1.k_feature_idx_
    print('The indices of best features SVM are: ', cols, '\n')
    str1 = ','.join(str(e) for e in cols)

    X_train = sfs1.transform(X_train)
    X_test = sfs1.transform(X_test)

    filename = 'saved_models/SFS_SVM.pkl'
    pickle.dump(sfs1, open(filename, 'wb'))
    train_acc = SVM_evaluation_procedure(SVC1, X_train, Y_train, X_test, Y_test)
    X_valid, Y_valid = SFS_validate.read_features(i)
    sfs1 = pickle.load(open(filename, 'rb'))
    X_valid = sfs1.transform(X_valid)
    val_acc = SFS_validate.validation_procedure_SVM(X_valid, Y_valid)
    return str1, train_acc, val_acc

def runFeatures_KNN(i, numberOfFeature, X_train, Y_train, X_test, Y_test):
    print("run feature method")
    print("Number of features Selected KNN : ", numberOfFeature)
    KNN = KNeighborsClassifier(n_neighbors = 8, p = 3)
    sfs1 = sfs(KNN, k_features = numberOfFeature, forward = True, floating = False, verbose = 2, scoring = 'accuracy', cv = 0, n_jobs = -1)
    sfs1.fit(X_train, Y_train)
    cols = sfs1.k_feature_idx_
    print('The indices of best features KNN are: ', cols, '\n')
    str1 = ','.join(str(e) for e in cols)

    X_train = sfs1.transform(X_train)
    X_test = sfs1.transform(X_test)

    filename = 'saved_models/SFS_KNN.pkl'
    pickle.dump(sfs1, open(filename, 'wb'))
    train_acc = KNN_evaluation_procedure(KNN, X_train, Y_train, X_test, Y_test)
    X_valid, Y_valid = SFS_validate.read_features(i)
    sfs1 = pickle.load(open(filename, 'rb'))
    X_valid = sfs1.transform(X_valid)
    val_acc = SFS_validate.validation_procedure_KNN(X_valid, Y_valid)
    return str1, train_acc, val_acc

sheet1 = wb.add_sheet('performance_sheet')

sheet1.write(0,0,'Number of features Selected')
sheet1.write(0,1,'Indices of Best Features')
sheet1.write(0,2,'Train Accuracy')
sheet1.write(0,3,'Validation Accuracy')
sheet1.write(0,4,'Model')
sheet1.write(0,5,'DataSet')

base_dir_train = ['/home/adi/Adi_research_data/BUZZ1/train/','/home/adi/Adi_research_data/BUZZ2/train/','/home/adi/Adi_research_data/BUZZ3/train/']
# base_dir_train = ['/home/adi/Adi_research_data/BUZZ3/train/']
base_dir_test = ['/home/adi/Adi_research_data/BUZZ1/test/','/home/adi/Adi_research_data/BUZZ2/test/','/home/adi/Adi_research_data/BUZZ3/test/']
# base_dir_test = ['/home/adi/Adi_research_data/BUZZ3/test/']
sub_dir_train = ['bee_train', 'noise_train', 'cricket_train', 'lawn_train']
# sub_dir_train = ['bee_train', 'noise_train', 'cricket_train']
sub_dir_test = ['bee_test', 'noise_test', 'cricket_test', 'lawn_test']
# sub_dir_test = ['bee_test', 'noise_test', 'cricket_test']
datasetNames = ["Buzz1","Buzz2","Buzz3"]

#for loop over Buzz1, Buzz2 and Buzz3 datasets
for i in range(2,3):
    #for loop over number of features to select
    for j in range(2,21):
        #for random forest
        X_train, Y_train, X_test, Y_test = read_features(base_dir_train[i], base_dir_test[i],sub_dir_train,sub_dir_test)
        indices,train_acc,val_acc = runFeatures_RF(i,j, X_train, Y_train, X_test, Y_test)
        sheet1.write(rowNum, 0, j)
        sheet1.write(rowNum, 1, indices)
        sheet1.write(rowNum, 2, train_acc)
        sheet1.write(rowNum, 3, val_acc)
        sheet1.write(rowNum, 4, "Random Forest")
        sheet1.write(rowNum, 5, datasetNames[i])
        rowNum = rowNum+1
        # for Logistic Regression
        indices,train_acc,val_acc = runFeatures_LR(i,j, X_train, Y_train, X_test, Y_test)
        sheet1.write(rowNum, 0, j)
        sheet1.write(rowNum, 1, indices)
        sheet1.write(rowNum, 2, train_acc)
        sheet1.write(rowNum, 3, val_acc)
        sheet1.write(rowNum, 4, "Logistic Regression")
        sheet1.write(rowNum, 5, datasetNames[i])
        rowNum = rowNum + 1
        #for SVM
        # indices,train_acc,val_acc = runFeatures_SVM(i,j, X_train, Y_train, X_test, Y_test)
        # sheet1.write(rowNum, 0, j)
        # sheet1.write(rowNum, 1, indices)
        # sheet1.write(rowNum, 2, train_acc)
        # sheet1.write(rowNum, 3, val_acc)
        # sheet1.write(rowNum, 4, "SVM")
        # sheet1.write(rowNum, 5, datasetNames[i])
        # rowNum = rowNum + 1
        #for KNN
        indices, train_acc,val_acc = runFeatures_KNN(i,j, X_train, Y_train, X_test, Y_test)
        sheet1.write(rowNum, 0, j)
        sheet1.write(rowNum, 1, indices)
        sheet1.write(rowNum, 2, train_acc)
        sheet1.write(rowNum, 3, val_acc)
        sheet1.write(rowNum, 4, "KNN")
        sheet1.write(rowNum, 5, datasetNames[i])
        rowNum = rowNum + 1
    wb.save("Buzz4_SFS_Overall.xls")
