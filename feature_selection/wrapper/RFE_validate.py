from __future__ import print_function
import os
import glob
import pickle
import numpy as np

from sklearn.metrics import confusion_matrix

base_dir = ['/home/adi/Adi_research_data/BUZZ1/out_of_sample_data_for_validation/','/home/adi/Adi_research_data/BUZZ2/out_of_sample_data_for_validation/','/home/adi/Adi_research_data/BUZZ3/out_of_sample_data_for_validation/']
sub_dir = ['bee_test', 'noise_test', 'cricket_test', 'lawn_test']


def read_features(i):
    D = []
    L = []

    for label, class_names in enumerate(sub_dir, start = 0):
        mvector_fft_path = os.path.join(base_dir[i],class_names, "pyaudio_features", "*.mvector.npy")
        all_files = glob.glob(mvector_fft_path)
        for f in all_files:
            value = np.load(f)
            D.append(value[:])
            L.append(label)
    return np.array(D),np.array(L)


X_test, Y_test = read_features(2)

filename = 'saved_models/RFE_RF.pkl'
rfe = pickle.load(open(filename, 'rb'))
X_test = rfe.transform(X_test)


def validation_procedure_RF(X_test, Y_test):
    print ('Starting with the validation procedure RF', '\n')

    filename = 'saved_models/RFE_RF_auto_Model.pkl'
    model = pickle.load(open(filename, 'rb'))

    predict = model.predict(X_test)

    print ('The confusion matrix is')
    cm = confusion_matrix(Y_test, predict)
    print (cm, '\n')

    diagonal_sum = 0
    for i in cm.diagonal():
        diagonal_sum += i

    confusion_accuracy = ((diagonal_sum)/float(cm.sum())) * 100
    print ('The validation accuracy of the classifier is ', confusion_accuracy, "%", '\n')
    return confusion_accuracy

def validation_procedure_LR(X_test, Y_test):
    print ('Starting with the validation procedure LR', '\n')

    filename = 'saved_models/RFE_LR_auto_Model.pkl'
    model = pickle.load(open(filename, 'rb'))

    predict = model.predict(X_test)

    print ('The confusion matrix is')
    cm = confusion_matrix(Y_test, predict)
    print (cm, '\n')

    diagonal_sum = 0
    for i in cm.diagonal():
        diagonal_sum += i

    confusion_accuracy = ((diagonal_sum)/float(cm.sum())) * 100
    print ('The validation accuracy of the classifier is ', confusion_accuracy, "%", '\n')
    return confusion_accuracy


def validation_procedure_SVM(X_test, Y_test):
    print ('Starting with the validation procedure SVM', '\n')

    filename = 'saved_models/RFE_SVM_auto_Model.pkl'
    model = pickle.load(open(filename, 'rb'))

    predict = model.predict(X_test)

    print ('The confusion matrix is')
    cm = confusion_matrix(Y_test, predict)
    print (cm, '\n')

    diagonal_sum = 0
    for i in cm.diagonal():
        diagonal_sum += i

    confusion_accuracy = ((diagonal_sum)/float(cm.sum())) * 100
    print ('The validation accuracy of the classifier is ', confusion_accuracy, "%", '\n')
    return confusion_accuracy
# validation_procedure_RF(X_test, Y_test)
