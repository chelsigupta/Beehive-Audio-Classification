import os
import glob
import pickle
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

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

pca = PCA(n_components = 'mle', svd_solver = 'full', random_state = 12)
pca.fit(X_train)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

filename = 'pickle_modified/PCA_Variance.pkl'
pickle.dump(pca, open(filename, 'wb'))


def KNN_evaluation_procedure(X_train, Y_train, X_test, Y_test):
    print ('Starting with the KNN procedure', '\n')

    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)

    model = KNeighborsClassifier(n_neighbors = 8, p = 3)
    model.fit(X_train, Y_train)

    filename = 'pickle_modified/PCA_Variance_KNN.pkl'
    pickle.dump(model, open(filename, 'wb'))

    predict = model.predict(X_test)

    print ('The confusion matrix is')
    cm = confusion_matrix(Y_test, predict)
    print (cm, '\n')

    diagonal_sum = 0
    for i in cm.diagonal():
        diagonal_sum += i

    confusion_accuracy = ((diagonal_sum)/float(cm.sum())) * 100
    print ('The accuracy of KNN classifier is ', confusion_accuracy, "%", '\n')


KNN_evaluation_procedure(X_train, Y_train, X_test, Y_test)


def RF_evaluation_procedure(X_train, Y_train, X_test, Y_test):
    print ('Starting with Random Forest procedure', '\n')

    model = RandomForestClassifier(n_estimators = 100, random_state = 1, max_features = 'log2')
    model.fit(X_train, Y_train)

    filename = 'pickle_modified/PCA_Variance_RF.pkl'
    pickle.dump(model, open(filename, 'wb'))

    predict = model.predict(X_test)

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

    model = LogisticRegression(penalty = 'l1', tol = 0.1, random_state = 12)
    model.fit(X_train, Y_train)

    filename = 'pickle_modified/PCA_Variance_LR.pkl'
    pickle.dump(model, open(filename, 'wb'))

    predict = model.predict(X_test)

    print ('The confusion matrix is')
    cm = confusion_matrix(Y_test, predict)
    print (cm, '\n')

    diagonal_sum = 0
    for i in cm.diagonal():
        diagonal_sum += i

    confusion_accuracy = ((diagonal_sum)/float(cm.sum())) * 100
    print ('The accuracy of Logistic Regression classifier is ', confusion_accuracy, "%", '\n')


LR_evaluation_procedure(X_train, Y_train, X_test, Y_test)


def OVR_SVM_evaluation_procedure(X_train ,Y_train, X_test, Y_test):
    print ('Starting with OneVsRest SVM procedure', '\n')

    model = OneVsRestClassifier(svm.SVC(kernel = 'poly', degree = 3, probability = True, random_state = 0))
    model.fit(X_train, Y_train)

    filename = 'pickle_modified/PCA_Variance_SVM.pkl'
    pickle.dump(model, open(filename, 'wb'))

    predict = model.predict(X_test)

    print ('The confusion matrix is')
    cm = confusion_matrix(Y_test, predict)
    print (cm, '\n')

    diagonal_sum = 0
    for i in cm.diagonal():
        diagonal_sum += i

    confusion_accuracy = ((diagonal_sum)/float(cm.sum())) * 100
    print ('The accuracy of OneVsRest SVM classifier is ', confusion_accuracy, "%", '\n')


OVR_SVM_evaluation_procedure(X_train, Y_train, X_test, Y_test)
