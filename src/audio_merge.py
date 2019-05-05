# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:02:37 2019

@author: macfa
"""
import librosa
import os
import soundfile as sf


audio_path = '../../100_download/separated_data/audio'
mix_path = '../../100_download/mixture_data/audio'
nparray_path = '../../100_download/separated_data/nparray'
SR = 16000
audio_num = 100
def audio_merge(nparray_path, audio_path, mix_path, audio_num):
    f = []
    for (dirpath, dirnames, filenames) in os.walk(nparray_path):
        f.extend(filenames)
        break
    
    f = f[:audio_num]
    progress = 0

    for i in range(len(f)):
        path1 = f[i]
        name1 = path1[:-12]
        path1 = name1 + '.wav'
        
        path2_count = i + 1
        if path2_count == len(f):
            break
        while filenames[path2_count][:11] not in path1:
            path2 = f[path2_count]
            name2 = path2[:-12]
            path2 = name2 + '.wav'
            
            ### load
            signal1, _ = librosa.load(audio_path + '/' + path1, sr=SR)
            signal2, _ = librosa.load(audio_path + '/' + path2, sr=SR)
            ### 3.2s
            signal1_slice = signal1[:3*SR]
            signal2_slice = signal2[:3*SR]
            ### normalize
            signal1_n2 = librosa.util.normalize(signal1_slice, norm=2)
            signal2_n2 = librosa.util.normalize(signal2_slice, norm=2)
            ### merge
            signal3 = signal1_n2 + signal2_n2
            ### write mix_wav
            try:
                os.makedirs(mix_path)
            except FileExistsError:
                pass
            name3 = name1 + '~' + name2 + '.wav'
            dir3 = mix_path + '/' + name3
            sf.write(dir3, signal3, samplerate=SR)
            
            progress += 1
            print("Progress: {0}-{1}/{2}".format(i,path2_count,len(f)))
#             ### feature extraction
#             signal3_mel = librosa.feature.melspectrogram(y=signal3, sr=SR, n_mels=N_MEL)
#             signal1_mel = librosa.feature.melspectrogram(y=signal1, sr=SR, n_mels=N_MEL)
#             signal2_mel = librosa.feature.melspectrogram(y=signal2, sr=SR, n_mels=N_MEL)
#             ### save h5py
            if path2_count == len(f)-1:
                break
            path2_count += 1

audio_merge(nparray_path, audio_path, mix_path, audio_num)       

