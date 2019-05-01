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

train_dataset = '../../new_dataset/audio_video/tr_set.hdf5'
val_dataset = '../../new_dataset/audio_video/val_set.hdf5'
test_dataset = '../../new_dataset/audio_video/test_set.hdf5'

batch_size = 10
epochs = 20

train_generator = data_generator(train_dataset, batch_size)
val_generator = data_generator(val_dataset, batch_size)





##V:video1 VV:video2
V1 = layers.Input(shape=(75,512), name='Video1_input')
V2 = layers.Conv1D(256, kernel_size=7,dilation_rate=1, padding='same',
                 activation='relu')(V1)
V3 = layers.BatchNormalization(axis=-1)(V2)
V4 = layers.Conv1D(256, kernel_size=5,dilation_rate=1, padding='same',
                 activation='relu')(V3)
V5 = layers.BatchNormalization(axis=-1)(V4)
V6 = layers.Conv1D(256, kernel_size=5,dilation_rate=2, padding='same',
                 activation='relu')(V5)
V7 = layers.BatchNormalization(axis=-1)(V6)
V8 = layers.Conv1D(256, kernel_size=5,dilation_rate=4, padding='same',
                 activation='relu')(V7)
V9 = layers.BatchNormalization(axis=-1)(V8)
V10 = layers.Conv1D(256, kernel_size=5,dilation_rate=8, padding='same',
                 activation='relu')(V9)
V11 = layers.BatchNormalization(axis=-1)(V10)
V12 = layers.Conv1D(256, kernel_size=5,dilation_rate=16, padding='same',
                 activation='relu')(V11)
V13 = layers.BatchNormalization(axis=-1)(V12)

# Inserts a dimension of 1 into a tensor's shape. (deprecated arguments)
V14 = Lambda(lambda x : tf.expand_dims(x, axis = -1))(V13)
# Resize images to size using nearest neighbor interpolation.
V15 = Lambda(lambda x: tf.image.resize_nearest_neighbor(x, size = (301, x.shape[-2])))(V14)
print("Video1 shape ", V15.shape)
VV1 = layers.Input(shape=(75,512), name='Video2_input')
VV2 = layers.Conv1D(256, kernel_size=7,dilation_rate=1, padding='same',
                 activation='relu')(VV1)
VV3 = layers.BatchNormalization(axis=-1)(VV2)
VV4 = layers.Conv1D(256, kernel_size=5,dilation_rate=1, padding='same',
                 activation='relu')(VV3)
VV5 = layers.BatchNormalization(axis=-1)(VV4)
VV6 = layers.Conv1D(256, kernel_size=5,dilation_rate=2, padding='same',
                 activation='relu')(VV5)
VV7 = layers.BatchNormalization(axis=-1)(VV6)
VV8 = layers.Conv1D(256, kernel_size=5,dilation_rate=4, padding='same',
                 activation='relu')(VV7)
VV9 = layers.BatchNormalization(axis=-1)(VV8)
VV10 = layers.Conv1D(256, kernel_size=5,dilation_rate=8, padding='same',
                 activation='relu')(VV9)
VV11 = layers.BatchNormalization(axis=-1)(VV10)
VV12 = layers.Conv1D(256, kernel_size=5,dilation_rate=16, padding='same',
                 activation='relu')(VV11)
VV13 = layers.BatchNormalization(axis=-1)(VV12)

# Inserts a dimension of 1 into a tensor's shape. (deprecated arguments)
VV14 = Lambda(lambda x : tf.expand_dims(x, axis = -1))(VV13)

# Resize images to size using nearest neighbor interpolation.
VV15 = Lambda(lambda x: tf.image.resize_nearest_neighbor(x, size = (301, x.shape[-2])))(VV14)
print("Video2 shape ", VV15.shape)

A1 = layers.Input(shape=(301,150,1), name='Audio_input')
A2 = layers.Conv2D(96, kernel_size=(1, 7),
                 dilation_rate=(1, 1),
                 activation='relu',padding='same')(A1)
A3 = layers.BatchNormalization(axis=-1)(A2)
A4 = layers.Conv2D(96, kernel_size=(7, 1),
                 dilation_rate=(1, 1),
                 activation='relu',padding='same',)(A3)
A5 = layers.BatchNormalization(axis=-1)(A4)
A6 = layers.Conv2D(96, kernel_size=(5, 5),
                 dilation_rate=(1, 1),
                 activation='relu',padding='same',)(A5)
A7 = layers.BatchNormalization(axis=-1)(A6)
A8 = layers.Conv2D(96, kernel_size=(5, 5),
                 dilation_rate=(2, 1),
                 activation='relu',padding='same',)(A7)
A9 = layers.BatchNormalization(axis=-1)(A8)
A10 = layers.Conv2D(96, kernel_size=(5, 5),
                 dilation_rate=(4, 1),
                 activation='relu',padding='same',)(A9)
A11 = layers.BatchNormalization(axis=-1)(A10)
A12 = layers.Conv2D(96, kernel_size=(5, 5),
                 dilation_rate=(8, 1),
                 activation='relu',padding='same',)(A11)
A13 = layers.BatchNormalization(axis=-1)(A12)
A14 = layers.Conv2D(96, kernel_size=(5, 5),
                 dilation_rate=(16, 1),
                 activation='relu',padding='same',)(A13)
A15 = layers.BatchNormalization(axis=-1)(A14)
A16 = layers.Conv2D(96, kernel_size=(5, 5),
                 dilation_rate=(32, 1),
                 activation='relu',padding='same',)(A15)
A17 = layers.BatchNormalization(axis=-1)(A16)
A18 = layers.Conv2D(96, kernel_size=(5, 5),
                 dilation_rate=(1, 1),
                 activation='relu',padding='same',)(A17)
A19 = layers.BatchNormalization(axis=-1)(A18)
A20 = layers.Conv2D(96, kernel_size=(5, 5),
                 dilation_rate=(2, 2),
                 activation='relu',padding='same',)(A19)
A21 = layers.BatchNormalization(axis=-1)(A20)
A22 = layers.Conv2D(96, kernel_size=(5, 5),
                 dilation_rate=(4, 4),
                 activation='relu',padding='same',)(A21)
A23 = layers.BatchNormalization(axis=-1)(A22)
A24 = layers.Conv2D(96, kernel_size=(5, 5),
                 dilation_rate=(8, 8),
                 activation='relu',padding='same',)(A23)
A25 = layers.BatchNormalization(axis=-1)(A24)
A26 = layers.Conv2D(96, kernel_size=(5, 5),
                 dilation_rate=(16, 16),
                 activation='relu',padding='same',)(A25)
A27 = layers.BatchNormalization(axis=-1)(A26)
A28 = layers.Conv2D(96, kernel_size=(5, 5),
                 dilation_rate=(32, 32),
                 activation='relu',padding='same',)(A27)
A29 = layers.BatchNormalization(axis=-1)(A28)
A30 = layers.Conv2D(8, kernel_size=(1, 1),
                 dilation_rate=(1, 1),
                 activation='relu',padding='same',)(A29)
A31 = layers.BatchNormalization(axis=-1)(A30)
print("Audio shape ", A31.shape)
## concate
fa = TimeDistributed(Flatten())(A31)
fv = TimeDistributed(Flatten())(V15)
fvv = TimeDistributed(Flatten())(VV15)
concated = concatenate([fa, fv, fvv], axis = 2)
print("concated shape ", concated.shape)

AV1 = concated
AV2 = layers.Bidirectional(LSTM(units = 64, return_sequences = True, activation = "tanh"))(AV1)
AV3 = layers.Dense(400)(AV2)
AV4 = layers.Dense(400)(AV3)
AV5 = layers.Dense(400)(AV4)
AV6 = layers.Dense(300,activation='sigmoid')(AV5)
#mask
mask = Reshape([2 , 301, 150])(AV6) ;print("mask ", mask.shape)
mask1 = Lambda(lambda x : x[:,0])(mask) ;print("mask 1 ", mask1.shape)
mask2 = Lambda(lambda x : x[:,1])(mask) ;print("mask 2 ", mask2.shape)
#insert a dimension to multiply
mask1 = Lambda(lambda x : tf.expand_dims(x, axis = -1))(mask1) ;print("mask 1 after insert dimiension", mask1.shape)
mask2 = Lambda(lambda x : tf.expand_dims(x, axis = -1))(mask2) ;print("mask 2 after insert dimiension", mask2.shape)
#multiply with original
spec1 = Lambda(lambda x : tf.multiply(x[0], x[1]), name = "multiply1")([A1, mask1])
print("spec1's shape", spec1.shape)
spec2 = Lambda(lambda x : tf.multiply(x[0], x[1]), name = "multiply2")([A1, mask2])
print("spec2's shape", spec2.shape)
AV = Model(inputs=[A1,V1,VV1],outputs=[spec1,spec2])

# es = EarlyStopping(monitor='loss', min_delta=0, patience= 30,
#                    verbose=1, mode='min', baseline=None, restore_best_weights= True)
ES = EarlyStopping(monitor='loss',patience=5)
# Reduce learning rate when a metric has stopped improving.
# rp = ReduceLROnPlateau(monitor=['loss'], factor=0.01, patience=5, verbose=1, mode='auto',
#                        min_delta=0.0001, cooldown=0, min_lr=0)
tb = TensorBoard(log_dir='./tb_logs', histogram_freq=0, batch_size= batch_size,
                 write_graph=True, write_grads=False, write_images=False,
                 embeddings_freq=0, embeddings_layer_names=None,
                 embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
AV.compile(loss='mean_squared_error', optimizer='Adam',metrics=['accuracy'])

#history = AV.fit([spec_mix,video_1,video_2],[spec_1,spec_2],
#         validation_split = 0.2,
#         batch_size = batch_size,
#         epochs = epochs,
#         callbacks = [ES,tb])

history = AV.fit_generator(generator= train_generator, validation_data = val_generator, validation_steps = 50/batch_size, epochs = 5, steps_per_epoch = 200/batch_size, verbose = 1, shuffle = True,
                  callbacks = [ES, tb])

#score = AV.evaluate([spec_mix,video_1,video_2], [spec_1,spec_2], verbose=1)
#print('Test loss & accuracy:', score)
# print('Test accuracy:', score[1])

model_dir = '../../model'

####

try:
    os.makedirs(model_dir)
except FileExistsError:
    pass
AV.save(model_dir + '/AV.h5')


import json
history_dir = '../../history'
try:
    os.makedirs(history_dir)
except FileExistsError:
    pass

# save json
with open(history_dir + '/history.json', 'w') as fp:
    r = json.dump(history.history, fp, indent=2)

# read json
with open('history.json', 'r') as fp:
    history = json.load(fp)
    print(history)

