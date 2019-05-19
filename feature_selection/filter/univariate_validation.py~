from __future__ import print_function
import os
import glob
import pickle
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

base_dir = '/home/chelsi/BUZZ2/validate/'
sub_dir = ['bee_test', 'noise_test', 'cricket_test']


def read_features():
    D = []
    L = []

    for label, class_names in enumerate(sub_dir, start = 0):
        mvector_fft_path = os.path.join(base_dir,class_names, "pyaudio_features", "*.mvector.npy")
        all_files = glob.glob(mvector_fft_path)
        for f in all_files:
            value = np.load(f)
            D.append(value[:])
            L.append(label)

    return np.array(D),np.array(L)


X_test, Y_test = read_features()

'''minmax_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
X_test = minmax_scaler.fit_transform(X_test)''' #uncomment only for CHI2

filename = 'saved_models/UNI_MUT_INFO.pkl'
transformer = pickle.load(open(filename, 'rb'))
X_test = transformer.transform(X_test)


def KNN_validation_procedure(X_test, Y_test):
    print ('Starting with KNN validation procedure', '\n')

    X_test = preprocessing.scale(X_test)

    filename = 'saved_models/UNI_MUT_INFO_KNN.pkl'
    model = pickle.load(open(filename, 'rb'))

    predict = model.predict(X_test)

    print ('The confusion matrix is')
    cm = confusion_matrix(Y_test, predict)
    print (cm, '\n')

    diagonal_sum = 0
    for i in cm.diagonal():
        diagonal_sum += i

    confusion_accuracy = ((diagonal_sum)/float(cm.sum())) * 100
    print ('The validation accuracy of KNN classifier is ', confusion_accuracy, "%", '\n')


KNN_validation_procedure(X_test, Y_test)


def RF_validation_procedure(X_test, Y_test):
    print ('Starting with Random Forest validation procedure', '\n')

    filename = 'saved_models/UNI_MUT_INFO_RF.pkl'
    model = pickle.load(open(filename, 'rb'))

    predict = model.predict(X_test)

    print ('The confusion matrix is')
    cm = confusion_matrix(Y_test, predict)
    print (cm, '\n')

    diagonal_sum = 0
    for i in cm.diagonal():
        diagonal_sum += i

    confusion_accuracy = ((diagonal_sum)/float(cm.sum())) * 100
    print ('The validation accuracy of Random Forest classifier is ', confusion_accuracy, "%", '\n')


RF_validation_procedure(X_test, Y_test)


def LR_validation_procedure(X_test, Y_test):
    print ('Starting with Logistic Regression validation procedure', '\n')

    filename = 'saved_models/UNI_MUT_INFO_LR.pkl'
    model = pickle.load(open(filename, 'rb'))

    predict = model.predict(X_test)

    print ('The confusion matrix is')
    cm = confusion_matrix(Y_test, predict)
    print (cm, '\n')

    diagonal_sum = 0
    for i in cm.diagonal():
        diagonal_sum += i

    confusion_accuracy = ((diagonal_sum)/float(cm.sum())) * 100
    print ('The validation accuracy of Logistic Regression classifier is ', confusion_accuracy, "%", '\n')


LR_validation_procedure(X_test, Y_test)


def OVR_SVM_validation_procedure(X_test, Y_test):
    print ('Starting with OneVsRest SVM validation procedure', '\n')

    filename = 'saved_models/UNI_MUT_INFO_OVR_SVM.pkl'
    model = pickle.load(open(filename, 'rb'))

    predict = model.predict(X_test)

    print ('The confusion matrix is')
    cm = confusion_matrix(Y_test, predict)
    print (cm, '\n')

    diagonal_sum = 0
    for i in cm.diagonal():
        diagonal_sum += i

    confusion_accuracy = ((diagonal_sum)/float(cm.sum())) * 100
    print ('The validation accuracy of OneVsRest SVM classifier is ', confusion_accuracy, "%", '\n')


OVR_SVM_validation_procedure(X_test, Y_test)
