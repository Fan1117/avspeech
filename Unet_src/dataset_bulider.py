# -*- coding: utf-8 -*-
"""
Created on Thu May 16 16:47:41 2019

@author: macfa
"""
import librosa
import os
import h5py
from config import PARAS
import numpy as np
mixture_path = '../../Unetdata/mixture_data/audio'
normalize_audio_path = '../../Unetdata/norm_data/audio'
h5py_path = '../../Unetdata/dataset'

SR = 16000
from mel_dealer import mel_converter
def frame_feature_extractor(signal, mel_converter=mel_converter):
    """
    Takes in new signals and create mel chunks 
    """
    S = mel_converter.signal_to_melspec(signal)
    print(S.shape)
    if not S.shape[0] % (2*PARAS.N_MEL) == 0:
        S = S[:-1 * (S.shape[0] % (2*PARAS.N_MEL))] # divide the mel spectrogram
        
    chunk_num = int(S.shape[0] / (2*PARAS.N_MEL))
    mel_chunks = np.split(S, chunk_num) # create 150 * 150 data frames
    return mel_chunks[0]

f = []
for (dirpath, dirnames, filenames) in os.walk(mixture_path):
    f.extend(filenames)
    break
    
try:
    os.makedirs(h5py_path)
except FileExistsError:
    pass
tr_name = h5py_path + '/' + 'tr_set.hdf5'
val_name = h5py_path + '/' + 'val_set.hdf5'
test_name = h5py_path + '/' + 'test_set.hdf5'
tr_dataset = h5py.File(tr_name, 'a')
val_dataset = h5py.File(val_name, 'a')
test_dataset = h5py.File(test_name, 'a')
for i in range(len(f)):
    file_mix = f[i]
    file_1, file_2 = file_mix.split('~')
    file_2 = file_2[:-4]
    print("{0}/{1}".format(i, len(f)))
    path_mix = mixture_path + '/' + file_mix
    path_1 = normalize_audio_path + '/' + file_1 + '.wav'
    path_2 = normalize_audio_path + '/' + file_2 + '.wav'
    signal_mix, _ = librosa.load(path_mix, sr=SR)
    signal_1, _ = librosa.load(path_1, sr=SR)
    signal_2, _ = librosa.load(path_2, sr=SR)
    mel_mix = frame_feature_extractor(signal_mix, mel_converter=mel_converter)
    mel_1 = frame_feature_extractor(signal_1, mel_converter=mel_converter)
    mel_2 = frame_feature_extractor(signal_2, mel_converter=mel_converter)

    if i < 2000:
        if i == 0:
            tr_dataset.create_dataset('spec_mix', shape=(2000, 256, 128), dtype=np.float32)
            tr_dataset.create_dataset('spec_1', shape=(2000, 256, 128), dtype=np.float32)
            tr_dataset.create_dataset('spec_2', shape=(2000, 256, 128), dtype=np.float32)

        tr_dataset['spec_mix'][i] = mel_mix
        tr_dataset['spec_1'][i] = mel_1
        tr_dataset['spec_2'][i] = mel_2

    elif i < 2500:
        if i == 2000:
            val_dataset.create_dataset('spec_mix', shape=(500, 256, 128), dtype=np.float32)
            val_dataset.create_dataset('spec_1', shape=(500, 256, 128), dtype=np.float32)
            val_dataset.create_dataset('spec_2', shape=(500, 256, 128), dtype=np.float32)

        val_dataset['spec_mix'][i-2000] = mel_mix
        val_dataset['spec_1'][i-2000] = mel_1
        val_dataset['spec_2'][i-2000] = mel_2

    elif i < 3000:

        if i == 2500:
            test_dataset.create_dataset('spec_mix', shape=(500, 256, 128), dtype=np.float32)
            test_dataset.create_dataset('spec_1', shape=(500, 256, 128), dtype=np.float32)
            test_dataset.create_dataset('spec_2', shape=(500, 256, 128), dtype=np.float32)

        test_dataset['spec_mix'][i-2500] = mel_mix
        test_dataset['spec_1'][i-2500] = mel_1
        test_dataset['spec_2'][i-2500] = mel_2

    else:
        break


tr_dataset.close()
val_dataset.close()
test_dataset.close()