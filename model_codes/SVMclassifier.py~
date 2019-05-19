from __future__ import print_function
import os
import glob
import pickle
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

base_dir_train = '/home/chelsi/BUZZ1/train/'
sub_dir_train = ['bee_train', 'noise_train', 'cricket_train']

base_dir_test = '/home/chelsi/BUZZ1/test/'
sub_dir_test = ['bee_test', 'noise_test', 'cricket_test']


def read_mother_vector():
    D_train = []
    L_train = []

    for label, class_names in enumerate(sub_dir_train, start = 0):
        mvector_fft_path = os.path.join(base_dir_train, class_names, "pyaudio_features", "*.mvector.npy")
        all_files = glob.glob(mvector_fft_path)
        for f in all_files:
            value = np.load(f)
            D_train.append(value[8:21])
            L_train.append(label)

    D_test = []
    L_test = []

    for label, class_names in enumerate(sub_dir_test, start = 0):
        mvector_fft_path = os.path.join(base_dir_test, class_names, "pyaudio_features", "*.mvector.npy")
        all_files = glob.glob(mvector_fft_path)
        for f in all_files:
            value = np.load(f)
            D_test.append(value[8:21])
            L_test.append(label)

    return np.array(D_train), np.array(L_train), np.array(D_test), np.array(L_test)


X_train, Y_train, X_test, Y_test = read_mother_vector()


def evaluation_procedure(X_train, Y_train, X_test, Y_test):
    print ('Starting with SVM procedure', '\n')

    model = SVC(kernel='poly', probability=True, random_state = 0)
    model.fit(X_train, Y_train)

    filename = 'saved_models/BUZZ1_SVM.pkl'
    pickle.dump(model, open(filename, 'wb'))
    
    predict = model.predict(X_test)

    print ('The confusion matrix is')
    cm = confusion_matrix(Y_test, predict)
    print (cm, '\n')

    diagonal_sum = 0
    for i in  cm.diagonal():
        diagonal_sum += i

    confusion_accuracy = ((diagonal_sum)/float(cm.sum())) * 100
    print ('The accuracy of SVM classifier is ', confusion_accuracy, "%", '\n')


evaluation_procedure(X_train ,Y_train, X_test, Y_test)
