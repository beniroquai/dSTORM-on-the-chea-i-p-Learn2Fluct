# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 09:00:20 2020

@author: diederichbenedict
"""
from keras.callbacks import ModelCheckpoint
from keras.layers.wrappers import TimeDistributed
from keras.layers import Dense,Flatten,Input,concatenate,Dot, Conv2D,Reshape, MaxPooling2D, UpSampling2D,Conv3DTranspose, ZeroPadding2D,Conv3D,Conv2DTranspose, BatchNormalization, Dropout
from keras.models import Model,Sequential
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.callbacks import TensorBoard
import numpy as np


def SOFI(Ntime=1, Nbatch=1, Nx=100, Ny=100, features=1, Nchannel=1):
    # create our model here
    kernel_size = 3
    nfilters_lstm = 16
    
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
    
    if(0):
        # convLSTM 2
        ConvLSTM_2= ConvLSTM2D(filters=nfilters_lstm, kernel_size=(kernel_size, kernel_size)
                           , data_format='channels_last'
                           , recurrent_activation='hard_sigmoid'
                           , activation='tanh'
                           , padding='same', return_sequences=False)(BatchNorm_1)
        
        BatchNorm_2 = BatchNormalization()(ConvLSTM_2)
    else: 
        BatchNorm_2 = BatchNorm_1
    # conv2D 
    Conv_1 = Conv2D(1, kernel_size, strides=(1, 1), padding='same', data_format='channels_last',  activation='relu')(BatchNorm_2)
        
    
    # define the output
    seq = Model(inputs=inputs, outputs=Conv_1, name='Model ')
    
    return seq
  
    
  