# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:40:05 2019

@author: macfa
"""
import numpy as np
import h5py
tr_path = '../../dataset/no_video/test_set.hdf5'
###
def create_gt_mask(vocal_spec, bg_spec):
    """
    Take in log spectrogram and return a mask map for TF bins
    1 if the vocal sound is dominated in the TF-bin, while 0 for not
    """
#    vocal_spec = vocal_spec.numpy()
#    bg_spec = bg_spec.numpy()
    return np.array(vocal_spec > bg_spec, dtype=np.float32)
def data_generator(dataset_path, batch_size):
    dataset = h5py.File(dataset_path, 'a')
    spec_mix = dataset['spec_mix']
    spec_mix = np.array(spec_mix)
    
    spec_1 = dataset['spec_1']
    spec_1 = np.array(spec_1)
    
    spec_2 = dataset['spec_2']
    spec_2 = np.array(spec_2)
    
    spec_1_mask = create_gt_mask(spec_1, spec_2)
    spec_2_mask = create_gt_mask(spec_2, spec_1)
    
    video_1 = dataset['video_1']
    video_1 = np.array(video_1)
    
    video_2 = dataset['video_2']
    video_2 = np.array(video_2)
    
    set_new = True
    while True:
        for i in range(spec_mix.shape[0]):
            if set_new:
                input_spec_mix = np.zeros((batch_size, 301, 150))
                output_spec_1 = np.zeros((batch_size, 301, 150))
                output_spec_2 = np.zeros((batch_size, 301, 150))
                input_face_1 = np.zeros((batch_size, 75, 512))
                input_face_2 = np.zeros((batch_size, 75, 512))
                batch_index = 0
                set_new = False
                
            input_spec_mix[batch_index,:,:] = spec_mix[i,:,:]
            output_spec_1[batch_index,:,:] = spec_1_mask[i,:,:]
            output_spec_2[batch_index,:,:] = spec_2_mask[i,:,:]
            input_face_1[batch_index,:,:] = video_1[i,:,:]
            input_face_2[batch_index,:,:] = video_2[i,:,:]
            batch_index += 1
            
            if batch_index == batch_size:
                set_new = True
                batch_index = 0
                input_spec_mix = np.expand_dims(input_spec_mix, axis = -1)
                output_spec_1 = np.expand_dims(output_spec_1, axis = -1)
                output_spec_2 = np.expand_dims(output_spec_2, axis = -1)
                #yield [input_spec_mix, input_face_1, input_face_2], [output_spec_1, output_spec_2]
                yield spec_1, spec_2
                    
tr = data_generator('../../data/tr_set.hdf5',1)


            
    