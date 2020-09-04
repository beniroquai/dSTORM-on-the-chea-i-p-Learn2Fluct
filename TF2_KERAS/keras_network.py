# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 09:00:20 2020

@author: diederichbenedict
"""

from tensorflow.keras.layers import Dense,Flatten,Input,concatenate,Dot, Conv2D,Reshape, MaxPooling2D, UpSampling2D,Conv3DTranspose, ZeroPadding2D,Conv3D,Conv2DTranspose, BatchNormalization, Dropout, ConvLSTM2D
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np


def SOFI(Ntime=1, Nbatch=1, Nx=100, Ny=100, features=1, Nchannel=1, upsample=2, Nfilterlstm=4, NKernelsizelstm=3, reshape=False):
    '''define model'''
    # create our model here
    kernel_size = NKernelsizelstm
    nfilters_lstm = Nfilterlstm
    input_shape = (Nx,Ny,1)

    # define an input layer (Ntime, Nbatch, Nx, Ny, Nchannel)
    inputs = Input(name='x_input', dtype='float32', batch_shape=(Nbatch,Ntime, (Nx//upsample), (Ny//upsample)))


    # convert the 1D vector into the appropriate 4D tensor (for Android)
    if(reshape):
        inputs_reshape = Reshape(target_shape=(Ntime,Nx//upsample,Ny//upsample), name='reshape_3d')(inputs)
        input_norm = inputs_reshape 
    
        # normalization of the input (necessary for TFLite -> compuational more efficient)
        #input_norm = input_norm - tf.reduce_min(inputs_reshape)#, axis=[1,2,3])
        #input_norm = input_norm/tf.reduce_max(input_norm) #, axis=[1,2,3])
        input_norm = Reshape(target_shape=(Ntime,Nx//upsample,Ny//upsample,1), name='reshape_4d')(input_norm)
        print('ATTENTION: We need to normalize over one batch only, higher numbers are not allowed due to Tensorflowjs!')
    else:
        input_norm = Reshape(target_shape=(Ntime,Nx//upsample,Ny//upsample,1), name='reshape_4d')(inputs)
                                               
   # temporal feature extractoin 
    ConvLSTM_1= ConvLSTM2D(filters=nfilters_lstm , kernel_size=(kernel_size, kernel_size)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation='tanh'
                       , padding='same'
                       , return_sequences=False
                       , name='convlstm2d_1')(input_norm)
    
    #BatchNorm_1 = BatchNormalization(name='batchnorm_1')(ConvLSTM_1)
    
    if(0):
        # convLSTM 2
        ConvLSTM_2= ConvLSTM2D(filters=nfilters_lstm, kernel_size=(kernel_size, kernel_size)
                           , data_format='channels_last'
                           , recurrent_activation='hard_sigmoid'
                           , activation='tanh'
                           , padding='same', return_sequences=False)(BatchNorm_1)
        
        BatchNorm_2 = BatchNormalization()(ConvLSTM_2)
    else: 
        BatchNorm_2 = ConvLSTM_1
    
    if(1):
        # Feature extraction 
        Conv_1 = Conv2D(16, kernel_size, strides=(1, 1), padding='same', data_format='channels_last',  activation='relu', name='conv_1')(BatchNorm_2)
        Conv_2 = UpSampling2D(size=(2,2), name='upsampling_1')(Conv_1)
        Conv_3 = Conv2D(1, kernel_size=kernel_size, padding = 'same')(Conv_2)

    else:
        # Feature extraction 
        #Conv_1 = Conv2D(upsample**2, kernel_size, strides=(1, 1), padding='same', data_format='channels_last',  activation='relu')(BatchNorm_2)
            
        # upsampling
        Conv_2 = tf.nn.depth_to_space(BatchNorm_2, upsample)
        
        # Feature extraction 
        Conv_3 = Conv2D(1, kernel_size, strides=(1, 1), padding='same', data_format='channels_last',  activation='relu')(Conv_2)
        
    # reshape it back into a 1D vector     
    if(reshape):
        Conv_3_reshape = Reshape((Nx*Ny,), name='reshape_2d_1d')(Conv_3)
    else:
        Conv_3_reshape = Conv_3
    
    # define the output
    seq = Model(inputs=inputs, outputs=Conv_3_reshape, name='learn2sofi')
   
    return seq
  
    
    
def SubpixelConv2D(input_shape, scale=4):
    """
    Keras layer to do subpixel convolution.
    NOTE: Tensorflow backend only. Uses tf.depth_to_space
    Ref:
        [1] Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
            Shi et Al.
            https://arxiv.org/abs/1609.05158
    :param input_shape: tensor shape, (batch, height, width, channel)
    :param scale: upsampling scale. Default=4
    :return:
    """

    #https://github.com/keras-team/keras/issues/5298
    # upsample using depth_to_space
    def subpixel_shape(input_shape):
        import tensorflow as tf
        dims = [input_shape[0],
                input_shape[1] * scale,
                input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        import tensorflow as tf #https://github.com/keras-team/keras/issues/5298
        return tf.nn.depth_to_space(x, scale, data_format='NHWC')

    import tensorflow as tf
    #https://github.com/keras-team/keras/issues/5088
    return Lambda(subpixel, output_shape=subpixel_shape, name='subpixel')
  

