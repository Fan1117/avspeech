# -*- coding: utf-8 -*-
"""
Created on Thu May 16 16:31:04 2019

@author: macfa
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import os
from config import PARAS
import warnings

warnings.filterwarnings('ignore')
audio_path = '../../100_download/separated_data/audio'
mixture_path = '../../Unetdata/mixture_data/audio'
normalize_audio_path = '../../Unetdata/norm_data/audio'


SR = 16000

audio_list = []
for (dirpath, dirnames, filenames) in os.walk(audio_path):
    if not dirnames:
        audio_list.extend(filenames)
        
from mel_dealer import mel_converter
def frame_feature_extractor(signal, mel_converter=mel_converter):
    """
    Takes in new signals and create mel chunks 
    """
    S = mel_converter.signal_to_melspec(signal)
    if not S.shape[0] % (2*PARAS.N_MEL) == 0:
        S = S[:-1 * (S.shape[0] % (2*PARAS.N_MEL))] # divide the mel spectrogram
        
    chunk_num = int(S.shape[0] / (2*PARAS.N_MEL))
    mel_chunks = np.split(S, chunk_num) # create 150 * 150 data frames
    return mel_chunks[0]

f = audio_list
progress = 0
for i in range(len(f)):
    path1 = f[i]
    name1 = path1[:-4]
    path1 = name1 + '.wav'

    path2_count = i + 1
    if path2_count == len(f):
        break
    while filenames[path2_count][:11] not in path1:
        path2 = f[path2_count]
        name2 = path2[:-4]
        path2 = name2 + '.wav'
        signal1, _ = librosa.load(audio_path + '/' + path1, sr=SR)
        signal2, _ = librosa.load(audio_path + '/' + path2, sr=SR)
                
        mel_spec_1 = frame_feature_extractor(signal1, mel_converter=mel_converter)
        mel_spec_2 = frame_feature_extractor(signal2, mel_converter=mel_converter)
        
        res_signal_1 = mel_converter.m(mel_spec_1, log=True, phase=None, transpose=True, audio_out=False)
        res_signal_2 = mel_converter.m(mel_spec_2, log=True, phase=None, transpose=True, audio_out=False)
        
        signal1_n2 = librosa.util.normalize(res_signal_1, norm=2)
        signal2_n2 = librosa.util.normalize(res_signal_2, norm=2)
        
        signal3 = signal1_n2 + signal2_n2
        
        try:
            os.makedirs(normalize_audio_path)
        except FileExistsError:
            pass
        dir1 = normalize_audio_path + '/' + name1 + '.wav'
        sf.write(dir1, signal1_n2, samplerate=SR)
        dir2 = normalize_audio_path + '/' + name2 + '.wav'
        sf.write(dir2, signal2_n2, samplerate=SR)
        
        try:
            os.makedirs(mixture_path)
        except FileExistsError:
            pass
        name3 = name1 + '~' + name2 + '.wav'
        dir3 = mixture_path + '/' + name3
        sf.write(dir3, signal3, samplerate=SR)
        
        if path2_count == len(f)-1:
            break
        path2_count += 1
        
        progress += 1
        print("Progress: {0}-{1}/{2}".format(i,path2_count,len(f)))