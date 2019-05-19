import os
import glob
import pickle
import numpy as np

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from mlxtend.feature_selection import SequentialFeatureSelector as sfs

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

KNN = KNeighborsClassifier(n_neighbors = 8, p = 3)
RF = RandomForestClassifier(n_estimators = 100, random_state = 1, max_features = 'log2')
LR = LogisticRegression(penalty = 'l1', tol = 0.1, random_state = 12)
SVC = SVC(kernel = 'linear', probability = True, random_state = 0)

sfs1 = sfs(SVC, k_features = 14, forward = True, floating = False, verbose = 2, scoring = 'accuracy', cv = 0, n_jobs = -1)
sfs1.fit(X_train, Y_train)

cols = sfs1.k_feature_idx_
print ('The indices of best features are: ', cols, '\n')

X_train = sfs1.transform(X_train)
X_test = sfs1.transform(X_test)

filename = 'saved_models/SFS_SVM.pkl'
pickle.dump(sfs1, open(filename, 'wb'))


def KNN_evaluation_procedure(X_train, Y_train, X_test, Y_test):
    print ('Starting with KNN procedure', '\n')

    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)

    KNN.fit(X_train, Y_train)

    filename = 'saved_models/SFS_KNN_Model.pkl'
    pickle.dump(KNN, open(filename, 'wb'))

    predict = KNN.predict(X_test)

    print ('The confusion matrix is')
    cm = confusion_matrix(Y_test, predict)
    print (cm, '\n')

    diagonal_sum = 0
    for i in  cm.diagonal():
        diagonal_sum += i

    confusion_accuracy = ((diagonal_sum)/float(cm.sum())) * 100
    print ('The accuracy of KNN classifier is ', confusion_accuracy,"%", '\n')


#KNN_evaluation_procedure(X_train, Y_train, X_test, Y_test)


def RF_evaluation_procedure(X_train, Y_train, X_test, Y_test):
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


#RF_evaluation_procedure(X_train, Y_train, X_test, Y_test)


def LR_evaluation_procedure(X_train, Y_train, X_test, Y_test):
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


#LR_evaluation_procedure(X_train, Y_train, X_test, Y_test)


def SVM_evaluation_procedure(X_train, Y_train, X_test, Y_test):
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


SVM_evaluation_procedure(X_train, Y_train, X_test, Y_test)
