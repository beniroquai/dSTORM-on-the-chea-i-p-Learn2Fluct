from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
from collections import OrderedDict
import logging

import tensorflow as tf
import tensorflow.contrib as contrib

from sofilstm import util
from sofilstm.layers import *

from sofilstm.convlstm import BasicConvLSTMCell

def sofi_decoder(x, y, keep_prob, phase, img_channels, truth_channels, features_root=16, filter_size=3, pool_size=2, summaries=True):
    """
    Creates a new convolutional unet for the given parametrization.
    
    :param x: input tensor, shape [?,Nx,Nx,img_channels]
    :param keep_prob: dropout probability tensor
    :param img_channels: number of channels in the input image
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """
    
    logging.info("features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{pool_size}".format(
                                                                                                           features=features_root,
                                                                                                           filter_size=filter_size,
                                                                                                         pool_size=pool_size))
    # Placeholder for the input image
    mysize = x.get_shape().as_list()
    Nx = mysize[1]
    Ny = mysize[2]
    ntimesteps = mysize[3] # for bbrevity
    x_image = x
    batch_size = mysize[0]
    #in_node = conv2d_bn_relu(x_image, filter_size, features_root, keep_prob, phase, 'conv2feature_roots')
    
    with tf.variable_scope('convlstm_layer'):
        ### ---- LSTM STUFF
        
        # make the order of the layers following the convention of the LSTM
        in_node_lstm = tf.transpose(tf.expand_dims(x_image,-1),[3,0,1,2,4])
        
        # Apply the convLSTM layer for all time-steps
        cell = BasicConvLSTMCell([Nx,Ny], features_root, [filter_size,filter_size])
        output, state = tf.nn.dynamic_rnn(cell, in_node_lstm, dtype=in_node_lstm.dtype, time_major=True)
        
        # make the order of the layers following the convention of the LSTM
        # https://github.com/leena201818/radioml/blob/master/experiment/lstm/ConvLSTM.py
        lstm_out = tf.transpose(output,[1,2,3,0,4])
        in_node = lstm_out[:,:,:,-1,:] # select the last time step as the one we want to use
    
        # Output Map
        print('Attention: We use sigmoid as last activation!')
        with tf.variable_scope("conv2d_1by1"):
            #output = conv2d(in_node, 1, truth_channels, keep_prob, 'conv2truth_channels')
            output_end = conv2d_bn_relu(x_image, 1, truth_channels, keep_prob, phase, 'conv2truth_channels')
            #output = conv2d_sigmoid(in_node, 1, truth_channels, keep_prob, 'conv2truth_channels')
            #output = deconv2d_bn_relu_res(in_node, 1, truth_channels, 1, keep_prob, phase, 'conv2truth_channels')


    #if True: #summaries
    # Will reccord the summary for all images
    logging.info("Record all Image-Summaries")
    tf.summary.image('output_image', get_image_summary(output_end))
    tf.summary.image('input_image', get_image_summary(tf.expand_dims(x[0,:,:,:],0)))
    tf.summary.image('groundtruth_image', get_image_summary(tf.expand_dims(y[0,:,:,:],0)))
    tf.summary.image('input_image_mean', get_image_summary(tf.expand_dims(tf.math.reduce_mean(x,axis=-1),0)))
    tf.summary.image('input_image_std', get_image_summary(tf.expand_dims(tf.math.reduce_mean(x,axis=-1),0)))
    
    return output_end

def get_image_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """
    
    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255
    
    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V
