#Run with python version 3
import os
import glob
import numpy as np
import librosa


def create_mother_vector(file_name):

    X, sample_rate = librosa.load(file_name)

    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)

    mother_vector = np.hstack([mfccs,chroma,mel,contrast,tonnetz])

    head, tail = os.path.split(file_name)
    add_path = os.path.join(head, "librosa_features")
    name_file,ext = os.path.splitext(tail)
    new_name = name_file + ".mvector"
    new_path_file = os.path.join(add_path, new_name)
    np.save(new_path_file, mother_vector)


def generating_mother_vector_files_bee():
    os.chdir("/home/chelsi/BUZZ1/train/bee_train/")
    for filename in glob.glob('*.wav'):
        create_mother_vector(filename)


def generating_mother_vector_files_noise():
    os.chdir("/home/chelsi/BUZZ1/train/noise_train/")
    for filename in glob.glob('*.wav'):
        create_mother_vector(filename)


def generating_mother_vector_files_cricket():
    os.chdir("/home/chelsi/BUZZ1/train/cricket_train/")
    for filename in glob.glob('*.wav'):
        create_mother_vector(filename)


generating_mother_vector_files_bee()
generating_mother_vector_files_noise()
generating_mother_vector_files_cricket()
