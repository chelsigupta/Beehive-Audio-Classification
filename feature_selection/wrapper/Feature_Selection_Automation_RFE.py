from __future__ import print_function
import os
import glob
import pickle
import numpy as np
import xlwt
from xlwt import Workbook
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix
## Recursive Feature Elimination
from feature_selection.wrapper import RFE_validate
# from feature_selection.wrapper.RFE import RF_evaluation_procedure, LR_evaluation_procedure, SVM_evaluation_procedure
# from feature_selection.wrapper.RFE import RF, LR


def RF_evaluation_procedure(RF, X_train, Y_train, X_test, Y_test):
    print ('Starting with Random Forest procedure', '\n')

    RF.fit(X_train, Y_train)

    filename = 'saved_models/RFE_RF_auto_Model.pkl'
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


# RF_evaluation_procedure(X_train, Y_train, X_test, Y_test)


def LR_evaluation_procedure(LR, X_train, Y_train, X_test, Y_test):
    print ('Starting with Logistic Regression procedure', '\n')

    LR.fit(X_train, Y_train)

    filename = 'saved_models/RFE_LR_auto_Model.pkl'
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


# LR_evaluation_procedure(X_train, Y_train, X_test, Y_test)


def SVM_evaluation_procedure(SVC, X_train, Y_train, X_test, Y_test):
    print ('Starting with the SVM procedure', '\n')

    SVC.fit(X_train, Y_train)

    filename = 'saved_models/RFE_SVM_auto_Model.pkl'
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

# Workbook is created
wb = Workbook()

rowNum = 1


def runFeatures_RF(i, numberOfFeature, X_train, Y_train, X_test, Y_test):
        print("Number of features Selected : ",numberOfFeature)
        RF = RandomForestClassifier(n_estimators=100, random_state=1, max_features='log2')
        rfe = RFE(estimator = RF, n_features_to_select = numberOfFeature, step = 1)
        rfe.fit(X_train, Y_train)
        cols = rfe.get_support(indices=True)
        print('The indices of best features are: ', cols, '\n')
        str1 = ','.join(str(e) for e in cols)

        X_train = rfe.transform(X_train)
        X_test = rfe.transform(X_test)

        filename = 'saved_models/RFE_RF_auto.pkl'
        pickle.dump(rfe, open(filename, 'wb'))
        train_acc = RF_evaluation_procedure(RF,X_train, Y_train, X_test, Y_test)
        X_valid, Y_valid = RFE_validate.read_features(i)
        rfe = pickle.load(open(filename, 'rb'))
        X_valid = rfe.transform(X_valid)
        val_acc = RFE_validate.validation_procedure_RF(X_valid, Y_valid)
        return str1,train_acc,val_acc


def runFeatures_LR(i, numberOfFeature, X_train, Y_train, X_test, Y_test):
        print("Number of features Selected : ",numberOfFeature)
        LR = LogisticRegression(penalty = 'l1', tol = 0.1, random_state = 12)
        rfe = RFE(estimator = LR, n_features_to_select = numberOfFeature, step = 1)
        rfe.fit(X_train, Y_train)
        cols = rfe.get_support(indices=True)
        print('The indices of best features are: ', cols, '\n')
        str1 = ','.join(str(e) for e in cols)

        X_train = rfe.transform(X_train)
        X_test = rfe.transform(X_test)

        filename = 'saved_models/RFE_LR_auto.pkl'
        pickle.dump(rfe, open(filename, 'wb'))
        train_acc = LR_evaluation_procedure(LR, X_train, Y_train, X_test, Y_test)
        X_valid, Y_valid = RFE_validate.read_features(i)
        print("value of i ",i)
        rfe = pickle.load(open(filename, 'rb'))
        X_valid = rfe.transform(X_valid)
        val_acc = RFE_validate.validation_procedure_LR(X_valid, Y_valid)
        return str1, train_acc, val_acc


def runFeatures_SVM(i, numberOfFeature, X_train, Y_train, X_test, Y_test):
        print("Number of features Selected : ",numberOfFeature)
        svc = SVC(kernel = "linear", C = 1, probability = True, random_state = 0)
        rfe = RFE(estimator = svc, n_features_to_select = numberOfFeature, step = 1)
        rfe.fit(X_train, Y_train)
        cols = rfe.get_support(indices=True)
        print('The indices of best features are: ', cols, '\n')
        str1 = ','.join(str(e) for e in cols)

        X_train = rfe.transform(X_train)
        X_test = rfe.transform(X_test)

        filename = 'saved_models/RFE_SVM_auto.pkl'
        pickle.dump(rfe, open(filename, 'wb'))
        train_acc = SVM_evaluation_procedure(svc,X_train, Y_train, X_test, Y_test)
        X_valid, Y_valid = RFE_validate.read_features(i)
        rfe = pickle.load(open(filename, 'rb'))
        X_valid = rfe.transform(X_valid)
        val_acc = RFE_validate.validation_procedure_SVM(X_valid, Y_valid)
        return str1, train_acc, val_acc

# add_sheet is used to create sheet.
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
# sub_dir_train = ['bee_train', 'noise_train', 'cricket_train']
# sub_dir_train = ['bee_train']
sub_dir_train = ['bee_train', 'noise_train', 'cricket_train','lawn_train']
# sub_dir_test = ['bee_test']
sub_dir_test = ['bee_test', 'noise_test', 'cricket_test', 'lawn_test']
datasetNames = ["Buzz1","Buzz2","Buzz3"]

#for loop over Buzz1, Buzz2 and Buzz3 datasets
for i in range(2,3):
    #for loop over number of features to select
    for j in range(1,35):
        #for random forest
        X_train, Y_train, X_test, Y_test = read_features(base_dir_train[i], base_dir_test[i],sub_dir_train,sub_dir_test)
        #number of features, indices of features,train acc, valid, acc,model,dataset
        indices,train_acc,val_acc = runFeatures_RF(i, j, X_train, Y_train, X_test, Y_test)
        sheet1.write(rowNum, 0, j)
        sheet1.write(rowNum, 1, indices)
        sheet1.write(rowNum, 2, train_acc)
        sheet1.write(rowNum, 3, val_acc)
        sheet1.write(rowNum, 4, "Random Forest")
        sheet1.write(rowNum, 5, datasetNames[i])
        rowNum = rowNum+1
        # for Logistic Regression
        indices,train_acc,val_acc = runFeatures_LR(i, j, X_train, Y_train, X_test, Y_test)
        sheet1.write(rowNum, 0, j)
        sheet1.write(rowNum, 1, indices)
        sheet1.write(rowNum, 2, train_acc)
        sheet1.write(rowNum, 3, val_acc)
        sheet1.write(rowNum, 4, "Logistic Regression")
        sheet1.write(rowNum, 5, datasetNames[i])
        rowNum = rowNum + 1
        #for SVM
        indices,train_acc,val_acc = runFeatures_SVM(i, j, X_train, Y_train, X_test, Y_test)
        sheet1.write(rowNum, 0, j)
        sheet1.write(rowNum, 1, indices)
        sheet1.write(rowNum, 2, train_acc)
        sheet1.write(rowNum, 3, val_acc)
        sheet1.write(rowNum, 4, "SVM")
        sheet1.write(rowNum, 5, datasetNames[i])
        rowNum = rowNum + 1
wb.save('Buzz4_RFE_OverAll.xls')
