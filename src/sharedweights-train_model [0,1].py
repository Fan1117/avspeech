from keras.layers import *
from keras import Model
from keras.models import Sequential
import tensorflow as tf
import numpy as np
import keras
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#config = tf.ConfigProto(device_count = {'GPU': 1}) 
#sess = tf.Session(config=config) 
#keras.backend.set_session(sess)
#from keras import backend as K
#K.tensorflow_backend._get_available_gpus()
import keras.layers as layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
import h5py
from load_dataset import data_generator
import os

train_dataset = '../../200_dataset/norm_audio_video/tr_set.hdf5'
val_dataset = '../../200_dataset/norm_audio_video/val_set.hdf5'
test_dataset = '../../200_dataset/norm_audio_video/test_set.hdf5'

batch_size = 10
epochs = 100

train_generator = data_generator(train_dataset, batch_size)
val_generator = data_generator(val_dataset, batch_size)

class FullModel():

    def __init__(self):

        self.conv1 = Conv1D(filters = 256, kernel_size = 7, padding = "same", dilation_rate = 1,
                      activation = "relu")
        self.bn1 = BatchNormalization(axis=-1)

        self.conv2 = Conv1D(filters = 256, kernel_size = 5, padding = "same", dilation_rate = 1,
                      activation = "relu")
        self.bn2 = BatchNormalization(axis=-1)

        self.conv3 = Conv1D(filters = 256, kernel_size = 5, padding = "same", dilation_rate = 2,
                      activation = "relu")
        self.bn3 = BatchNormalization(axis=-1)

        self.conv4 = Conv1D(filters = 256, kernel_size = 5, padding = "same", dilation_rate = 4,
                      activation = "relu")
        self.bn4 = BatchNormalization(axis=-1)

        self.conv5 = Conv1D(filters = 256, kernel_size = 5, padding = "same", dilation_rate = 8,
                      activation = "relu")
        self.bn5 = BatchNormalization(axis=-1)

        self.conv6 = Conv1D(filters = 256, kernel_size = 5, padding = "same", dilation_rate = 16,
                      activation = "relu")
        self.bn6 = BatchNormalization(axis=-1)
        
# Inserts a dimension of 1 into a tensor's shape. (deprecated arguments)
        self.conv7 = Lambda(lambda x : tf.expand_dims(x, axis = -1))
  
# Resize images to size using nearest neighbor interpolation.
        self.conv8 = Lambda(lambda x: tf.image.resize_nearest_neighbor(x, size = (301, x.shape[-2])))

    def allmodel(self):
        
        video1 = layers.Input(shape=(75, 512)); print("Video_input", video1.shape) 
        video2 = layers.Input(shape=(75, 512)); print("Video_input", video2.shape) 

        stream_1 = self.conv1(video1)
        stream_1 = self.bn1(stream_1)
        stream_1 = self.conv2(stream_1)
        stream_1 = self.bn2(stream_1)
        stream_1 = self.conv3(stream_1)
        stream_1 = self.bn3(stream_1)
        stream_1 = self.conv4(stream_1)
        stream_1 = self.bn4(stream_1)
        stream_1 = self.conv5(stream_1)
        stream_1 = self.bn5(stream_1)
        stream_1 = self.conv6(stream_1)
        stream_1 = self.bn6(stream_1)
        stream_1 = self.conv7(stream_1) 
        video_stream_1 = self.conv8(stream_1)

        stream_2 = self.conv1(video2)
        stream_2 = self.bn1(stream_2)
        stream_2 = self.conv2(stream_2)
        stream_2 = self.bn2(stream_2)
        stream_2 = self.conv3(stream_2)
        stream_2 = self.bn3(stream_2)
        stream_2 = self.conv4(stream_2)
        stream_2 = self.bn4(stream_2)
        stream_2 = self.conv5(stream_2)
        stream_2 = self.bn5(stream_2)
        stream_2 = self.conv6(stream_2)
        stream_2 = self.bn6(stream_2)
        stream_2 = self.conv7(stream_2)
        video_stream_2 = self.conv8(stream_2)


        # input audio(mixed spectrogram)
        A1 = layers.Input(shape=(301,150,1), name='Audio_input')
        A2 = layers.Conv2D(96, kernel_size=(1, 7),
                         dilation_rate=(1, 1),
                         activation='relu',padding='same')(A1)
        A3 = layers.BatchNormalization(axis=-1)(A2)
        A4 = layers.Conv2D(96, kernel_size=(7, 1),
                         dilation_rate=(1, 1),
                         activation='relu', padding='same',)(A3)
        A5 = layers.BatchNormalization(axis=-1)(A4)
        A6 = layers.Conv2D(96, kernel_size=(5, 5),
                         dilation_rate=(1, 1),
                         activation='relu', padding='same',)(A5)
        A7 = layers.BatchNormalization(axis=-1)(A6)
        A8 = layers.Conv2D(96, kernel_size=(5, 5),
                         dilation_rate=(2, 1),
                         activation='relu', padding='same',)(A7)
        A9 = layers.BatchNormalization(axis=-1)(A8)
        A10 = layers.Conv2D(96, kernel_size=(5, 5),
                         dilation_rate=(4, 1),
                         activation='relu', padding='same',)(A9)
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
        Audio_stream = layers.BatchNormalization(axis=-1)(A30)
        print("input audio shape ", Audio_stream.shape)

        ## concate audio and video vector
        fa = TimeDistributed(Flatten())(Audio_stream)
        fv = TimeDistributed(Flatten())(video_stream_1)
        fvv = TimeDistributed(Flatten())(video_stream_2)
        concated = concatenate([fa, fv, fvv], axis = 2)
        print("concated tensor shape ", concated.shape)


        ## feed into a BLSTM
        AuVi1 = concated
        AuVi2 = layers.Bidirectional(LSTM(units = 64, return_sequences = True, activation = "tanh"))(AuVi1)
        AuVi3 = layers.Dense(400)(AuVi2)
        AuVi4 = layers.Dense(600)(AuVi3)
        AuVi5 = layers.Dense(600)(AuVi4)
        AuVi6 = layers.Dense(300,activation='sigmoid')(AuVi5)

        #generate mask
        mask = Reshape([2 , 301, 150])(AuVi6) ;print("mask ", mask.shape)
        mask1 = Lambda(lambda x : abs(x[:,0,:,:])/np.max(abs(x[:,0,:,:])))(mask) ;print("mask 1 ", mask1.shape)
        mask2 = Lambda(lambda x : abs(x[:,1,:,:])/np.max(abs(x[:,1,:,:])))(mask) ;print("mask 2 ", mask2.shape)
        # mask1 = Lambda(lambda x : x[:,0])(mask) ;print("mask 1 ", mask1.shape)
        # mask2 = Lambda(lambda x : x[:,1])(mask) ;print("mask 2 ", mask2.shape)

        #insert a dimension to multiply
        mask1 = Lambda(lambda x : tf.expand_dims(x, axis = -1))(mask1) ;print("mask 1 after insert dimiension", mask1.shape)
        mask2 = Lambda(lambda x : tf.expand_dims(x, axis = -1))(mask2) ;print("mask 2 after insert dimiension", mask2.shape)

        #multiply with mixed spectrogram (audio input of the network)
        spec1 = Lambda(lambda x : tf.multiply(x[0], x[1]), name = "multiply1")([A1, mask1])
        print("spec1's shape", spec1.shape)
        spec2 = Lambda(lambda x : tf.multiply(x[0], x[1]), name = "multiply2")([A1, mask2])
        print("spec2's shape", spec2.shape)
        model=Model(inputs=[A1,video1,video2],outputs=[spec1,spec2])
        return model

 
model= FullModel()  
AuVi=model.allmodel()    
# es = EarlyStopping(monitor='loss', min_delta=0, patience= 30,
#                    verbose=1, mode='min', baseline=None, restore_best_weights= True)
ES = EarlyStopping(monitor='loss',patience=5)
# Reduce learning rate when a metric has stopped improving.
# rp = ReduceLROnPlateau(monitor=['loss'], factor=0.01, patience=5, verbose=1, mode='auto',
#                        min_delta=0.0001, cooldown=0, min_lr=0)
try:
    os.makedirs('../../logs')
except FileExistsError:
    pass
tb = TensorBoard(log_dir='../../logs/200sw_tb_logs', histogram_freq=0, batch_size= batch_size,
                 write_graph=True, write_grads=False, write_images=False,
                 embeddings_freq=0, embeddings_layer_names=None,
                 embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
AuVi.compile(loss='mean_squared_error', optimizer='Adam',metrics=['accuracy'])

#history = AuVi.fit([spec_mix,video_1,video_2],[spec_1,spec_2],
#         validation_split = 0.2,
#         batch_size = batch_size,
#         epochs = epochs,
#         callbacks = [ES,tb])

history = AuVi.fit_generator(generator= train_generator, validation_data = val_generator, validation_steps = 500/batch_size, epochs = epochs, steps_per_epoch = 2000/batch_size, verbose = 1, shuffle = True,
                  callbacks = [ES, tb])

#score = AuVi.evaluate([spec_mix,video_1,video_2], [spec_1,spec_2], verbose=1)
#print('Test loss & accuracy:', score)
# print('Test accuracy:', score[1])

model_dir = '../../200sw_model'

####

try:
    os.makedirs(model_dir)
except FileExistsError:
    pass
AuVi.save(model_dir + '/AV_200sw.h5')


import json
history_dir = '../../200sw_history'
try:
    os.makedirs(history_dir)
except FileExistsError:
    pass

# save json
with open(history_dir + '/history_200sw.json', 'w') as fp:
    r = json.dump(history.history, fp, indent=2)

# read json
with open(history_dir + '/history_200sw.json', 'r') as fp:
    history = json.load(fp)
    print(history)


