# -*- coding: utf-8 -*-
"""
Created on Mon May 13 21:32:49 2019

@author: macfa
"""
tr_path = '../Dataset/train.h5'
val_path = '../Dataset/valid.h5'
test_path = '../Dataset/test.h5'

batch_size = 50
################################DATA
from keras.layers import Lambda
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from config import PARAS

from data_loader import torch_dataset_loader
train_loader = torch_dataset_loader(tr_path, batch_size, True, PARAS.kwargs)
validation_loader = torch_dataset_loader(val_path, batch_size, False, PARAS.kwargs)
test_loader = torch_dataset_loader(test_path, batch_size, False, PARAS.kwargs)

def data_generator(data_loader):
  while True:
    for index, data_item in enumerate(data_loader):
        yield np.expand_dims(np.array(data_item['mix']),-1), np.expand_dims(np.array(data_item['target']),-1)
        
train_generator = data_generator(train_loader) 
val_generator = data_generator(validation_loader) 
##############################DATA
##############################Model
import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import backend as K

import tensorflow as tf

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)




inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

#outputs = Lambda(lambda x: tf.multiply(x[0], x[1]), name = 'multiply')([inputs, mask])

from mel_dealer import mel_converter
import librosa

def res(train_loader, validation_loader, test_loader, num):
    ### generator for model 
    def data_generator(data_loader):
      while True:
        for index, data_item in enumerate(data_loader):
            yield np.expand_dims(np.array(data_item['mix']),-1), np.expand_dims(np.array(data_item['target']),-1)
            
    test_generator = data_generator(test_loader)
    X_test, y_test = next(test_generator)
    ### origin dataset
    for index, data_item in enumerate(test_loader):
        if index == 0 :
            break
    
    vocal = data_item['vocal'][num]
    mix = data_item['mix'][num]
    bg = data_item['bg'][num]
    target = data_item['target'][num]
    
    predict_model = Model(inputs=[inputs], outputs=[outputs])
    predict_model.load_weights('./model/unet_mask.h5')
    pre_mask = predict_model.predict(X_test)
    mix_amplitude = librosa.db_to_power(X_test[num,:,:,0])
    
    plt.figure()
    plt.imshow(mix, aspect='auto', origin='lower')
    plt.tight_layout()
    plt.show()
    
    
    plt.figure()
    plt.imshow(vocal, aspect='auto', origin='lower')
    plt.tight_layout()
    plt.show()
    
    
    pre_spec = np.array(mix_amplitude * pre_mask[num,:,:,0], dtype=np.float32)
    plt.figure()
    plt.imshow(librosa.power_to_db(pre_spec), aspect='auto', origin='lower')
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    plt.imshow(pre_mask[num,:,:,0], aspect='auto', origin='lower')
    plt.tight_layout()
    plt.show()
    
    mix_signal = mel_converter.m(mix.numpy(), log=True, audio_out=True)
    groudtruth_signal = mel_converter.m(vocal.numpy(), log=True, audio_out=True)
    pre_signal = mel_converter.m(librosa.power_to_db(pre_spec), log=True, audio_out=True)
    
    return mix_signal, groudtruth_signal, pre_signal
    
mix_signal, groudtruth_signal, pre_signal = res(train_loader, validation_loader, test_loader, 0)
    
