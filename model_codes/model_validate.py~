from __future__ import print_function
import os
import glob
import pickle
import numpy as np

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

base_dir = '/home/chelsi/BUZZ1/validate/'
sub_dir = ['bee_test', 'noise_test', 'cricket_test']


def read_features():
    D = []
    L = []

    for label, class_names in enumerate(sub_dir, start = 0):
        mvector_fft_path = os.path.join(base_dir, class_names, "pyaudio_features", "*.mvector.npy")
        all_files = glob.glob(mvector_fft_path)
        for f in all_files:
            value = np.load(f)
            D.append(value[8:21])
            L.append(label)

    return np.array(D), np.array(L)


X_test, Y_test = read_features()

#X_test = preprocessing.scale(X_test) #uncomment only for KNN


def evaluation_procedure(X_test, Y_test):
    print ('Starting with validation procedure', '\n')

    filename = 'saved_models/BUZZ1_RF.pkl'
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


evaluation_procedure(X_test, Y_test)
