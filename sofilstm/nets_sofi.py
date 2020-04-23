from __future__ import print_function, division, absolute_import, unicode_literals
import logging

import tensorflow as tf
import keras

import sofilstm.layers as layers
import sofilstm.util as util
import sofilstm.convlstmcell as convlstmExperimental
import sofilstm.subpixel as subpixel

from keras.layers import BatchNormalization, Conv2D

def sofi_decoder(x, y, keep_prob, phase, img_channels, truth_channels, hidden_num=4, kernel_size=3, pool_size=2, summaries=True, Ntime=30):
    """
    Creates a new convolutional unet for the given parametrization.
    
    :param x: input tensor, shape [?,Nx,Nx,img_channels]
    :param keep_prob: dropout probability tensor
    :param img_channels: number of channels in the input image
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param kernel_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """
    
    # Placeholder for the input image
    batch_size = x.get_shape()[0]
    Nx = tf.shape(x)[1]
    Ny = tf.shape(x)[2]
    Ntime = x.get_shape().as_list()[-1]
    x_image = tf.identity(x, 'x_input')
    #in_node = conv2d_bn_relu(x_image, kernel_size, features_root, keep_prob, phase, 'conv2feature_roots')
    input_norm = x_image - tf.reshape(tf.reduce_min(x_image, axis=[1,2,3]), [batch_size,1,1,1])
    input_norm = input_norm/tf.reshape(tf.reduce_max(input_norm, axis=[1,2,3]), [batch_size,1,1,1])
    
    in_node_lstm = tf.expand_dims(input_norm ,-1) #tf.expand_dims(input_norm,-1)
    
    p_input = in_node_lstm#   tf.placeholder(tf.float32, [batch_size, height, width, Ntime, channel])
    
    p_input_list = tf.split(p_input,Ntime,3)
    p_input_list = [tf.squeeze(p_input_, [3]) for p_input_ in p_input_list]

    cell = convlstmExperimental.ConvLSTMCell(hidden_num) # hidden_num 1
    state = cell.zero_state(batch_size, Nx, Ny)
    #state = tf.truncated_normal(shape=[batch_size, Nx, Ny, hidden_num_1*2], stddev=.5)        
    with tf.variable_scope("ConvLSTM") as scope: # as BasicLSTMCell
        for i, p_input_ in enumerate(p_input_list):
            print('Concat timestep: '+str(i))
            if i > 0: 
                scope.reuse_variables()
            # ConvCell takes Tensor with size [batch_size, height, width, channel].
            t_output, state = cell(p_input_, state)
            
    #BatchNorm_1 = BatchNormalization()(t_output)
    BatchNorm_1 = t_output
     
    # conv2D 
    Conv_0 = Conv2D(Ntime, kernel_size, strides=(1, 1), padding='same', data_format='channels_last',  activation='relu')(BatchNorm_1)
    Conv_1 = Conv2D(4, kernel_size, strides=(1, 1), padding='same', data_format='channels_last',  activation='relu')(Conv_0)
    Conv_2 = tf.depth_to_space(Conv_1, 2)
    #Conv_2 = subpixel.PS(Conv_1, 2, color=False)
    #Conv_2 = SubpixelConv2d(scale=2, n_out_channels=None, act=tf.nn.relu)(Conv_1)
    
    
    output_std = tf.expand_dims(tf.math.reduce_std(x,axis=-1),axis=-1)
    output_mean = tf.expand_dims(tf.math.reduce_mean(x,axis=-1),axis=-1)
    output_end = Conv_2#+output_mean
    output_raw = tf.identity(output_end, 'output_raw') # just to preserve the name
    
    psf_size = 9
    psf_sigma = 2
    psf_heatmap = util.matlab_style_gauss2D(shape = (psf_size,psf_size),sigma=psf_sigma)
    gfilter = tf.reshape(psf_heatmap, [psf_size, psf_size, 1, 1])
    output_end_psf = tf.nn.conv2d(output_raw, gfilter, [1, 1, 1, 1], padding="SAME")
    print('We add an additional convolution to the output map!')
    



    #if True: #summaries
    # Will reccord the summary for all images
    logging.info("Record all Image-Summaries")
    tf.summary.image('output_image', get_image_summary(output_end))
    tf.summary.image('output_image_psf', get_image_summary(output_end_psf))
    tf.summary.image('input_image', get_image_summary(tf.expand_dims(x[0,:,:,:],0)))
    tf.summary.image('groundtruth_image', get_image_summary(tf.expand_dims(y[0,:,:,:],0)))
    tf.summary.image('input_image_mean', get_image_summary(tf.expand_dims(tf.expand_dims(tf.math.reduce_mean(x[0,:,:,:],axis=-1),axis=0),axis=-1),0))
    tf.summary.image('input_image_std', get_image_summary(tf.expand_dims(tf.expand_dims(tf.math.reduce_std(x[0,:,:,:],axis=-1),axis=0),axis=-1),0))
       
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