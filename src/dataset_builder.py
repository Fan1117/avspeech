# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:59:34 2019

@author: macfa
"""
import librosa
import os
import h5py
import numpy as np
from config import PARAS
from IPython.display import Audio


class MelConverter:

    def __init__(self, sample_rate=PARAS.SR,
                 n_fft=PARAS.N_FFT,
                 #hop_length=PARAS.N_FFT // 4,
                 hop_length=160,
                 n_mel_freqs=PARAS.N_MEL,
                 freq_min_hz=0, freq_max_hz=None):

        self._SAMPLE_RATE = sample_rate
        self._N_FFT = n_fft
        self._HOP_LENGTH = hop_length
        self._N_MEL_FREQS = n_mel_freqs
        self._FREQ_MIN_HZ = freq_min_hz
        self._FREQ_MAX_HZ = freq_max_hz

        self._MEL_FILTER = librosa.filters.mel(
            sr=self._SAMPLE_RATE,
            n_fft=self._N_FFT,
            n_mels=self._N_MEL_FREQS,
            fmin=self._FREQ_MIN_HZ,
            fmax=self._FREQ_MAX_HZ)

    def signal_to_melspec(self, audio_signal, log=True, get_phase=False, transpose=False):
        D = librosa.core.stft(audio_signal, n_fft=self._N_FFT, hop_length=self._HOP_LENGTH)
        magnitude, phase = librosa.core.magphase(D)
        mel_spectrogram = np.dot(self._MEL_FILTER, magnitude)
        mel_spectrogram = mel_spectrogram ** 2

        if log:
            mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        if transpose:
            mel_spectrogram = mel_spectrogram.T
        if get_phase:
            return mel_spectrogram, phase
        else:
            return mel_spectrogram

    def melspec_to_audio(self, mel_spectrogram, log=True, phase=None, transpose=False, audio_out=True):
        if transpose:
            mel_spectrogram = mel_spectrogram.T
        if log:
            mel_spectrogram = librosa.db_to_power(mel_spectrogram)

        mel_spectrogram = mel_spectrogram ** 0.5

        magnitude = np.dot(np.linalg.pinv(self._MEL_FILTER), mel_spectrogram)

        if phase is not None:
            inverted_signal = librosa.istft(magnitude * phase, hop_length=self._HOP_LENGTH)
        else:
            inverted_signal = griffin_lim(magnitude, self._N_FFT, self._HOP_LENGTH, n_iterations=10)

        if audio_out:
            return Audio(inverted_signal, rate=self._SAMPLE_RATE)
        else:
            return inverted_signal


def griffin_lim(magnitude, n_fft, hop_length, n_iterations):
    """Iterative algorithm for phase retrival from a magnitude spectrogram."""
    phase_angle = np.pi * np.random.rand(*magnitude.shape)
    D = invert_magnitude_phase(magnitude, phase_angle)
    signal = librosa.istft(D, hop_length=hop_length)

    for i in range(n_iterations):
        D = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
        _, phase = librosa.magphase(D)
        phase_angle = np.angle(phase)

        D = invert_magnitude_phase(magnitude, phase_angle)
        signal = librosa.istft(D, hop_length=hop_length)

    return signal


def invert_magnitude_phase(magnitude, phase_angle):
    phase = np.cos(phase_angle) + 1.j * np.sin(phase_angle)
    return magnitude * phase
#########################################################################
SR = 16000
separated_path = '../../new_download/separated_data/audio'
mix_path = '../../new_download/mixture_data/audio'
h5py_path = '../../new_dataset/audio_video'
face_embeddings_folder = '../../new_download/separated_data/nparray'

def spec_extraction(mix_path, separated_path, face_embeddings_folder, h5py_path, sample_num):
    
    ### face_embeddings
    faces_list = []
    for (dirpath, dirnames, v_filenames) in os.walk(face_embeddings_folder):
        if not dirnames:
            faces_list.extend(v_filenames)
            break
        
    
    

    
    
    
    
    
    ### spec
    f = []
    converter = MelConverter()
    for (dirpath, dirnames, filenames) in os.walk(mix_path):
        f.extend(filenames)
        break

    f = f[:sample_num]

    
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
    #############################################
    for i in range(len(f)):
        
        file_mix = f[i]
        file_1, file_2 = file_mix.split('~')
        file_2 = file_1[:-4]
        nparray_1 = file_1 + '_facenet' + '.npy'
        nparray_2 = file_2 + '_facenet' + '.npy'
        print(nparray_1)
        print(faces_list)
        if nparray_1 not in faces_list:
            continue
        if nparray_2 not in faces_list:
            continue
        print("{0}/{1}".format(i, len(f)))
        ### audio path
        path_mix = mix_path + '/' + file_mix
        path_1 = separated_path + '/' + file_1 + '.wav'
        path_2 = separated_path + '/' + file_2 + '.wav'
        ### video path
        video_path_1 = face_embeddings_folder + '/' + nparray_1
        video_path_2 = face_embeddings_folder + '/' + nparray_2
        ### video signal
        video_1 = np.load(video_path_1)
        video_2 = np.load(video_path_2)### 75 * 512
    
    
#    
    
    
    ##############################################
#    for i in range(len(f)):
#        print("{0}/{1}".format(i, len(f)))
#        file_mix = f[i]
#        file_1, file_2 = file_mix.split('~')
#        
#        path_mix = mix_path + '/' + file_mix
#        path_1 = separated_path + '/' + file_1 + '.wav'
#        path_2 = separated_path + '/' + file_2

        
        
        
        signal_mix, _ = librosa.load(path_mix, sr=SR)
        signal_1, _ = librosa.load(path_1, sr=SR)
        signal_2, _ = librosa.load(path_2, sr=SR)

        ### convert
        mel_mix = converter.signal_to_melspec(signal_mix[:3*SR], transpose=True)
        mel_1 = converter.signal_to_melspec(signal_1[:3*SR], transpose=True)
        mel_2 = converter.signal_to_melspec(signal_2[:3*SR], transpose=True)
        
        if i < 200:
            if i == 0:
                tr_dataset.create_dataset('spec_mix', shape=(200, 301, 150), dtype=np.float32)
                tr_dataset.create_dataset('spec_1', shape=(200, 301, 150), dtype=np.float32)
                tr_dataset.create_dataset('spec_2', shape=(200, 301, 150), dtype=np.float32)
                ### video
                tr_dataset.create_dataset('video_1', shape=(200, 75, 512), dtype=np.float32)
                tr_dataset.create_dataset('video_2', shape=(200, 75, 512), dtype=np.float32)
                
                
            tr_dataset['spec_mix'][i] = mel_mix
            tr_dataset['spec_1'][i] = mel_1
            tr_dataset['spec_2'][i] = mel_2
            ### video
            tr_dataset['video_1'][i] = video_1
            tr_dataset['video_2'][i] = video_2
            
        elif i < 250:
            if i == 200:
                val_dataset.create_dataset('spec_mix', shape=(50, 301, 150), dtype=np.float32)
                val_dataset.create_dataset('spec_1', shape=(50, 301, 150), dtype=np.float32)
                val_dataset.create_dataset('spec_2', shape=(50, 301, 150), dtype=np.float32)
                ### video
                val_dataset.create_dataset('video_1', shape=(50, 75, 512), dtype=np.float32)
                val_dataset.create_dataset('video_2', shape=(50, 75, 512), dtype=np.float32)
                
                
            val_dataset['spec_mix'][i-200] = mel_mix
            val_dataset['spec_1'][i-200] = mel_1
            val_dataset['spec_2'][i-200] = mel_2
            ### video
            val_dataset['video_1'][i-200] = video_1
            val_dataset['video_2'][i-200] = video_2
            
#            if i == 2000:
#                val_dataset.create_dataset('spec_mix', shape=(500, 301, 150), dtype=np.float32)
#                val_dataset.create_dataset('spec_1', shape=(500, 301, 150), dtype=np.float32)
#                val_dataset.create_dataset('spec_2', shape=(500, 301, 150), dtype=np.float32)
#            val_dataset['spec_mix'][i-2000] = mel_mix
#            val_dataset['spec_1'][i-2000] = mel_1
#            val_dataset['spec_2'][i-2000] = mel_2      
            
        elif i < 300:

            if i == 250:
                test_dataset.create_dataset('spec_mix', shape=(50, 301, 150), dtype=np.float32)
                test_dataset.create_dataset('spec_1', shape=(50, 301, 150), dtype=np.float32)
                test_dataset.create_dataset('spec_2', shape=(50, 301, 150), dtype=np.float32)
                ### video
                test_dataset.create_dataset('video_1', shape=(50, 75, 512), dtype=np.float32)
                test_dataset.create_dataset('video_2', shape=(50, 75, 512), dtype=np.float32)
                
                
            test_dataset['spec_mix'][i-250] = mel_mix
            test_dataset['spec_1'][i-250] = mel_1
            test_dataset['spec_2'][i-250] = mel_2
            ### video
            test_dataset['video_1'][i-250] = video_1
            test_dataset['video_2'][i-250] = video_2            

            
#            if i == 250:
#                test_dataset.create_dataset('spec_mix', shape=(500, 301, 150), dtype=np.float32)
#                test_dataset.create_dataset('spec_1', shape=(500, 301, 150), dtype=np.float32)
#                test_dataset.create_dataset('spec_2', shape=(500, 301, 150), dtype=np.float32)
#            test_dataset['spec_mix'][i-2500] = mel_mix
#            test_dataset['spec_1'][i-2500] = mel_1
#            test_dataset['spec_2'][i-2500] = mel_2
            
        else:
            break
        
        
    tr_dataset.close()
    val_dataset.close()
    test_dataset.close()
            

spec_extraction(mix_path, separated_path, face_embeddings_folder, h5py_path, 3000)
#spec_extraction(mix_path, separated_path, h5py_path, 3000)         
            
            
        
        
        