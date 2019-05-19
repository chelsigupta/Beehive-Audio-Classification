import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

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

pca = PCA(n_components = 34)
pca.fit(X_train)

print ('The variance of each principal component is: ', pca.explained_variance_ratio_, '\n')

#plotting principal components and their variances
plt.xticks(np.arange(0,34,2))
plt.xlabel('principal component')
plt.ylabel('variance ratio')
plt.plot(pca.explained_variance_)
plt.savefig('pca_variance_plot.png')
