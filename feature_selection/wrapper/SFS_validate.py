import os
import glob
import pickle
import numpy as np

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

base_dir = '/home/chelsi/BUZZ2/validate/'
sub_dir = ['bee_test', 'noise_test', 'cricket_test']


def read_features():
    D = []
    L = []

    for label, class_names in enumerate(sub_dir, start = 0):
        mvector_fft_path = os.path.join(base_dir, class_names, "pyaudio_features", "*.mvector.npy")
        all_files = glob.glob(mvector_fft_path)
        for f in all_files:
            value = np.load(f)
            D.append(value[:])
            L.append(label)

    return np.array(D), np.array(L)


X_test, Y_test = read_features()

filename = 'saved_models/SFS_SVM.pkl'
sfs1 = pickle.load(open(filename, 'rb'))
X_test = sfs1.transform(X_test)

#X_test = preprocessing.scale(X_test) #uncomment only for KNN


def validation_procedure(X_test, Y_test):
    print ('Starting with the validation procedure', '\n')

    filename = 'saved_models/SFS_SVM_Model.pkl'
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


validation_procedure(X_test, Y_test)
