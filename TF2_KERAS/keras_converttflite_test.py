# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 22:51:10 2020

@author: diederichbenedict
"""

from tensorflow.keras.layers.wrappers import TimeDistributed
from tensorflow.keras.layers import Dense,Flatten,Input,concatenate,Dot, Conv2D,Reshape, MaxPooling2D, UpSampling2D,Conv3DTranspose, ZeroPadding2D,Conv3D,Conv2DTranspose, BatchNormalization, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Input, LSTM, Dropout, Dense, Concatenate, TimeDistributed, Permute
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers.convolutional_recurrent import ConvLSTM2D
from tensorflow.keras.callbacks import TensorBoard
import keras.losses as losses
import numpy as np

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



# create our model here
kernel_size = 3
nfilters_lstm = 16

Nbatch=1
Ntime=10
Nx=32
Ny=32
Nchannel=1


# define an input layer (Ntime, Nbatch, Nx, Ny, Nchannel)
inputs = Input(name='x_input', dtype='float32',batch_shape=(Nbatch, Ntime, Nx, Ny, Nchannel)) 

'''define model'''
# convLSTM 1
ConvLSTM_1= ConvLSTM2D(filters=nfilters_lstm , kernel_size=(kernel_size, kernel_size)
                   , data_format='channels_last'
                   , recurrent_activation='hard_sigmoid'
                   , activation='tanh'
                   , padding='same', return_sequences=False)(inputs)

BatchNorm_1 = BatchNormalization()(ConvLSTM_1)

# conv2D 
Conv_1 = Conv2D(1, kernel_size, strides=(1, 1), padding='same', data_format='channels_last',  activation='relu')(BatchNorm_1)
    
# define the output
model = Model(inputs=inputs, outputs=Conv_1, name='ModelNN')

model.compile(loss=losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])


import tensorflow as tf
MODEL_PATH = 'test1.hdf5'
converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(MODEL_PATH)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open("3x9.tflite", "wb").write(tflite_model)


import tensorflow as tf
MODEL_PATH = 'test1.hdf5'
model = tf.keras.models.load_model(MODEL_PATH)
model.summary()
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open("3x9.tflite", "wb").write(tflite_model)