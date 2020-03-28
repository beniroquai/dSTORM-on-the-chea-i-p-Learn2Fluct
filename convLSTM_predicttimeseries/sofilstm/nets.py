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

from sofilstm.convlstm import my_convlstm, convlstm_bidrectional

def unet_decoder(x, keep_prob, phase, img_channels, truth_channels, ntimesteps=9, layers=3, conv_times=3, features_root=16, filter_size=3, pool_size=2, summaries=True, is_bidrectional = True, is_skipconnections=True):
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
    
    logging.info("Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{pool_size}".format(layers=layers,
                                                                                                           features=features_root,
                                                                                                           filter_size=filter_size,
                                                                                                           pool_size=pool_size))
    # Placeholder for the input image
    Nx = tf.shape(x)[2]
    Ny = tf.shape(x)[3]
    ntimesteps = ntimesteps # for bbrevity
    if(False):
        x_image = tf.reshape(x, tf.stack([-1,Nx,Ny,img_channels]))
    else:
        x_image = x
    batch_size = tf.shape(x_image)[0]//ntimesteps

    pools = OrderedDict()  # pooling layers
    deconvs = OrderedDict()  # deconvolution layer
    dw_h_convs = OrderedDict()  # down-side convs
    up_h_convs = OrderedDict()  # up-side convs

    # conv the input image to desired feature maps
    in_node = conv2d_bn_relu(x_image, filter_size, features_root, keep_prob, phase, 'conv2feature_roots')

    # Down layers
    for layer in range(0, layers):
        features = 2**layer*features_root
        print('Featuresize @ Down-Layer '+str(layer) + ' is: '+str(features))
        with tf.variable_scope('down_layer_' + str(layer)):
            for conv_iter in range(0, conv_times):
                scope = 'conv_bn_relu_{}'.format(conv_iter)
                conv = conv2d_bn_relu(in_node, filter_size, features, keep_prob, phase, scope)    
                in_node = conv

            # store the intermediate result per layer
            dw_h_convs[layer] = in_node
            
            # down sampling
            if layer < layers-1:
                with tf.variable_scope('pooling'):
                    pools[layer] = max_pool(dw_h_convs[layer], pool_size)
                    in_node = pools[layer]

    n_features_convlstm = features
    insize_convlstm_m = tf.shape(in_node)[1]
    insize_convlstm_n = tf.shape(in_node)[2]
    in_node = dw_h_convs[layers-1]

    if(False):
        # the ordering of the channels is as follows now: [nbatch, Nx, Nx, nchannels, ntimesteps]
        # here i simply need to transpose the input to have:  [nbatch, ntimesteps, Nx, Nx, nchannels]
        nchannels = tf.shape(in_node)[-1]
        noutputchannels = features_root*2**conv_times
        print(noutputchannels)
        in_node = tf.reshape(in_node, [batch_size, ntimesteps, insize_convlstm_m, insize_convlstm_n, n_features_convlstm]) # shape: [nbatch, ntime, nx, ny, nf]
        lstm_out = convlstm(in_node, Nx=insize_convlstm_m, Ny=insize_convlstm_n, nchannels=1024, noutputchannels = n_features_convlstm, name = "conv_lstm_cell_1")
        print(lstm_out)
        # invert the permutation [nbatch, ntimesteps, Nx, Nx, nchannels] -> 
        nchannels = tf.shape(in_node)[-1]
        in_node = tf.reshape(lstm_out, [batch_size*ntimesteps, 8, 8, n_features_convlstm])
    
    with tf.variable_scope('convlstm_layer'):
        ### ---- LSTM STUFF
        #TODO: Make adaptive filtersize here!!
        in_node = tf.reshape(in_node, [batch_size, ntimesteps, insize_convlstm_m, insize_convlstm_n, n_features_convlstm]) # shape: [nbatch, ntime, nx, ny, nf]
        lstm_input = tf.transpose(in_node, [1, 0, 2, 3, 4])  # to fit the time_major
        if(is_bidrectional):
            lstm_out = convlstm_bidrectional(input=lstm_input, name='convlstm1', output_channel=n_features_convlstm, kernel_size=3, keep_prob=keep_prob, train=phase)
        else:
            lstm_out = my_convlstm(input=lstm_input, name='convlstm1', output_channel=n_features_convlstm, kernel_size=3, keep_prob=keep_prob, train=phase)
        in_node = tf.reshape(lstm_out[-1], [batch_size*ntimesteps, insize_convlstm_m, insize_convlstm_n, n_features_convlstm])
    
    # Up layers
    for layer in range(layers-2, -1, -1):
        features = 2**(layer+1)*features_root
        with tf.variable_scope('up_layer_' + str(layer)):
            with tf.variable_scope('unsample_concat_layer'):
                # number of features = lower layer's number of features
                print('Featuresize @ Up-Layer ' + str(layer) + ' is: ' + str(features))
                #h_deconv = deconv2d_bn_relu(in_node, filter_size, features//2, pool_size, keep_prob, phase, 'unsample_layer')
                h_deconv = deconv2d_bn_relu_res(in_node, filter_size, features//2, 1, keep_prob, phase, 'unsample_layer')
                if(is_skipconnections):
                    h_deconv_concat = concat(dw_h_convs[layer], h_deconv)
                else:
                    h_deconv_concat = h_deconv
                deconvs[layer] = h_deconv_concat
                in_node = h_deconv_concat

            for conv_iter in range(0, conv_times):
                scope = 'conv_bn_relu_{}'.format(conv_iter)
                conv = conv2d_bn_relu(in_node, filter_size, features//2, keep_prob, phase, scope)    
                in_node = conv            

            up_h_convs[layer] = in_node

    in_node = up_h_convs[0]

    if(False):
        # Output with residual
        with tf.variable_scope("conv2d_1by1"):
            output = conv2d(in_node, 1, truth_channels, keep_prob, 'conv2truth_channels')
            up_h_convs["out"] = in_node
    else:
        # Output Map
        print('Attention: We use sigmoid as last activation!')
        with tf.variable_scope("conv2d_1by1"):
            output = conv2d(in_node, 1, truth_channels, keep_prob, 'conv2truth_channels')
            up_h_convs["out"] = in_node

    
    #if True: #summaries
    # Will reccord the summary for all images
    logging.info("Record all Image-Summaries")
    tf.summary.image('output_image', get_image_summary(output))
    tf.summary.image('input_image', get_image_summary(x))

    for k in pools.keys():
        tf.summary.image('summary_pool_%02d'%k, get_image_summary(pools[k]))

    for k in deconvs.keys():
        tf.summary.image('summary_deconv_concat_%02d'%k, get_image_summary(deconvs[k]))

    for k in dw_h_convs.keys():
        tf.summary.histogram("dw_convolution_%02d"%k + '/activations', dw_h_convs[k])

    for k in up_h_convs.keys():
        tf.summary.histogram("up_convolution_%s"%k + '/activations', up_h_convs[k])

    return output

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