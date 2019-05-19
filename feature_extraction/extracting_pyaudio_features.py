#Run with python version 2
import os
import glob
import numpy as np

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction


def create_mother_vector(file_name):
    #This function extracts 34 features for each .wav file.

    [Fs, x] = audioBasicIO.readAudioFile(file_name);
    F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.020*Fs, 0.020*Fs);
    features = np.mean(F.T, axis=0) 
     
    #mother vector is the vector which contains all 34 features.
    mother_vector = np.hstack([features])

    head, tail = os.path.split(file_name)
    add_path = os.path.join(head, "pyaudio_features")
    name_file, ext = os.path.splitext(tail)
    new_name = name_file + ".mvector"
    new_path_file = os.path.join(add_path, new_name)
    np.save(new_path_file, mother_vector)


def generating_mother_vector_files_bee():
    #change the names of directories for train, test and validate accordingly.
    os.chdir("/home/chelsi/BUZZ1/train/bee_train/")
    for filename in glob.glob('*.wav'):
        create_mother_vector(filename)


def generating_mother_vector_files_noise():
    #change the names of directories for train, test and validate accordingly.
    os.chdir("/home/chelsi/BUZZ1/train/noise_train/")
    for filename in glob.glob('*.wav'):
        create_mother_vector(filename)


def generating_mother_vector_files_cricket():
    #change the names of directories for train, test and validate accordingly.
    os.chdir("/home/chelsi/BUZZ1/train/cricket_train/")
    for filename in glob.glob('*.wav'):
        create_mother_vector(filename)


#generating_mother_vector_files_bee()
#generating_mother_vector_files_noise()
#generating_mother_vector_files_cricket()
