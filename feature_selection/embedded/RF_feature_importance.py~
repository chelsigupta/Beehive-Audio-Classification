import os
import glob
import numpy as np
from itertools import cycle

import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import matplotlib.colors

import scipy
import scipy.io
from scipy.io import wavfile
from scipy import interp

from sklearn.ensemble import RandomForestClassifier

base_dir_train = '/home/chelsi/BUZZ2/train/'
sub_dir_train = ['bee_train', 'noise_train', 'cricket_train']


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

    return np.array(D_train), np.array(L_train)


X_train, Y_train = read_features()


def evaluation_procedure(X_train, Y_train):

    forest = RandomForestClassifier(n_estimators = 100, random_state = 1, max_features = 'log2')
    forest.fit(X_train, Y_train)    

    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis = 0)
    indices = np.argsort(importances)[::-1]

    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    plt.figure()
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Importance Scores")
    plt.bar(range(X_train.shape[1]), importances, color = "r", yerr = std, align = "center")
    plt.xticks(range(0,X_train.shape[1],2))
    plt.xlim([-1, X_train.shape[1]])
    plt.savefig('RF_feature_importance.png')


evaluation_procedure(X_train, Y_train)
