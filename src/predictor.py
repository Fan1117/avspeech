# -*- coding: utf-8 -*-
"""
Created on Wed May  1 09:55:41 2019

@author: macfa
"""
from keras.layers import *
from keras import Model
from keras.models import Sequential
import tensorflow as tf
import numpy as np
import keras
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
import keras.layers as layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
import h5py
from load_dataset import data_generator
import os
####
import numpy as np
import librosa
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
#############################################################
from keras.models import load_model
from load_dataset import data_generator
import soundfile as sf
import os
##############################################################
model_dir = '../../model'
model_path = '../../model/AV_30.h5'
test_dataset = '../../new_dataset/audio_video/test_set.hdf5'
test_generator = data_generator(test_dataset, 1)
[input_spec_mix, input_face_1, input_face_2], [output_spec_1, output_spec_2] = next(test_generator)
converter = MelConverter()
AV = load_model(model_path, custom_objects={'tf':tf})
#############################################################
def spectrogram_separator(input_spec_mix, input_face_1, input_face_2, output_spec_1, output_spec_2):
    ### predict
    mask_pre_1, mask_pre_2 = AV.predict([input_spec_mix, input_face_1, input_face_2])
    pre_spec_1 = input_spec_mix * mask_pre_1
    pre_spec_2 = input_spec_mix * mask_pre_2
    ###
    spec_mix = np.squeeze(input_spec_mix, axis=-1)
    spec_mix = np.squeeze(spec_mix, axis=0)
    
    pre_spec_1 = np.squeeze(pre_spec_1, axis=-1)
    pre_spec_1 = np.squeeze(pre_spec_1, axis=0)
    
    pre_spec_2 = np.squeeze(pre_spec_2, axis=-1)
    pre_spec_2 = np.squeeze(pre_spec_2, axis=0)
    ### target
    output_spec_1 = np.squeeze(output_spec_1, axis=-1)
    output_spec_1 = np.squeeze(output_spec_1, axis=0)
    
    output_spec_2 = np.squeeze(output_spec_2, axis=-1)
    output_spec_2 = np.squeeze(output_spec_2, axis=0)
    
    try:
        os.makedirs('../../np_res/')
    except FileExistsError:
        pass   
    np.save('../../np_res/mix.npy', spec_mix)
    np.save('../../np_res/pre_1.npy', pre_spec_1)
    np.save('../../np_res/pre_2.npy', pre_spec_2)
    np.save('../../np_res/tar_1.npy', output_spec_1)
    np.save('../../np_res/tar_2.npy', output_spec_2)
    
    sigal_pre_1 = converter.melspec_to_audio(pre_spec_1, transpose=True, audio_out=False)
    sigal_pre_2 = converter.melspec_to_audio(pre_spec_2, transpose=True, audio_out=False)
    
    sigal_target_1 = converter.melspec_to_audio(output_spec_1, transpose=True, audio_out=False)
    sigal_target_2 = converter.melspec_to_audio(output_spec_2, transpose=True, audio_out=False)
    try:
        os.makedirs('../../res')
    except FileExistsError:
        pass
    sf.write('../../res/pre_1.wav', sigal_pre_1, samplerate=PARAS.SR)
    sf.write('../../res/pre_2.wav', sigal_pre_2, samplerate=PARAS.SR)
    sf.write('../../res/tar_1.wav', sigal_target_1, samplerate=PARAS.SR)
    sf.write('../../res/tar_2.wav', sigal_target_2, samplerate=PARAS.SR)
    
spectrogram_separator(input_spec_mix, input_face_1, input_face_2, output_spec_1, output_spec_2)