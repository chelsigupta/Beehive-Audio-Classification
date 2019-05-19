from __future__ import print_function
import os
import glob
import pickle
import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix

base_dir_train = '/home/chelsi/BUZZ2/train/'
sub_dir_train = ['bee_train', 'noise_train', 'cricket_train']

base_dir_test = '/home/chelsi/BUZZ2/test/'
sub_dir_test = ['bee_test', 'noise_test', 'cricket_test']


def read_features():
    D_train = []
    L_train = []

    for label, class_names in enumerate(sub_dir_train, start = 0):
        mvector_fft_path = os.path.join(base_dir_train, class_names, "pyaudio_features", "*.mvector.npy")
        all_files = glob.glob(mvector_fft_path)
        for f in all_files:
            value = np.load(f)
            D_train.append(value[:])
            L_train.append(label)

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


X_train, Y_train, X_test, Y_test = read_features()

# Create the RFE object and rank each feature
RF = RandomForestClassifier(n_estimators = 100, random_state = 1, max_features = 'log2')
LR = LogisticRegression(penalty = 'l1', tol = 0.1, random_state = 12)
SVC = SVC(kernel = "linear", C = 1, probability = True, random_state = 0)

rfe = RFE(estimator = SVC, n_features_to_select = 14, step = 1)
rfe.fit(X_train, Y_train)

cols = rfe.get_support(indices = True)
print ('The indices of best features are: ', cols, '\n')

X_train = rfe.transform(X_train)
X_test = rfe.transform(X_test)

filename = 'saved_models/RFE_RF.pkl'
pickle.dump(rfe, open(filename, 'wb'))


def RF_evaluation_procedure(X_train, Y_train, X_test, Y_test):
    print ('Starting with Random Forest procedure', '\n')

    RF.fit(X_train, Y_train)

    filename = 'saved_models/RFE_RF_Model.pkl'
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


RF_evaluation_procedure(X_train, Y_train, X_test, Y_test)


def LR_evaluation_procedure(X_train, Y_train, X_test, Y_test):
    print ('Starting with Logistic Regression procedure', '\n')

    LR.fit(X_train, Y_train)

    filename = 'saved_models/RFE_LR_Model.pkl'
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


#LR_evaluation_procedure(X_train, Y_train, X_test, Y_test)


def SVM_evaluation_procedure(X_train, Y_train, X_test, Y_test):
    print ('Starting with the SVM procedure', '\n')

    SVC.fit(X_train, Y_train)

    filename = 'saved_models/RFE_SVM_Model.pkl'
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


#SVM_evaluation_procedure(X_train, Y_train, X_test, Y_test)
