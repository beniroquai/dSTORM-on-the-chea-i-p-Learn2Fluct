from __future__ import print_function, division, absolute_import, unicode_literals
import logging

import tensorflow as tf
import keras

import sofilstm.layers as layers
import sofilstm.util as util
import sofilstm.convlstmcell as convlstmExperimental
import sofilstm.subpixel as subpixel

from keras.layers import BatchNormalization, Conv2D, Dropout

def sofi_decoder(x, y, keep_prob, phase, hidden_num=4, kernel_size=3, summaries=True, Ntime=30, is_normalize=False, is_peephole=False):
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
    
    # normalization of the input (necessary for TFLite -> compuational more efficient)
    input_norm = x_image - tf.reshape(tf.reduce_min(x_image, axis=[1,2,3]), [batch_size,1,1,1])
    input_norm = input_norm/tf.reshape(tf.reduce_max(input_norm, axis=[1,2,3]), [batch_size,1,1,1])

    if(1):
        # need to add the channel information
        in_node_lstm = tf.expand_dims(input_norm ,-1)
    
        # split time steps
        p_input_list = tf.split(in_node_lstm,Ntime,3)
        p_input_list = [tf.squeeze(p_input_, [3]) for p_input_ in p_input_list]
    
        # create the convlstm2d cell
        cell = convlstmExperimental.ConvLSTMCell(hidden_num, is_normalize=is_normalize, is_peephole=is_peephole) # hidden_num 1
        state = cell.zero_state(batch_size, Nx, Ny)
        #state = tf.truncated_normal(shape=[batch_size, Nx, Ny, hidden_num_1*2], stddev=.5)        
        
        with tf.variable_scope("ConvLSTM") as scope: # as BasicLSTMCell
            for i, p_input_ in enumerate(p_input_list):
                print('Concat timestep: '+str(i))
                if i > 0: 
                    scope.reuse_variables()
                # ConvCell takes Tensor with size [batch_size, height, width, channel].
                Conv_t, state = cell(p_input_, state)
         
        # Feature extraction
        #Conv_t_BN = BatchNormalization()(Conv_t)
        #b = tf.cond(tf.equal(a, tf.constant(True)), lambda: tf.constant(10), lambda: tf.constant(0))
        Conv_t_do = Conv_t#tf.nn.dropout(Conv_t, keep_prob)
        Conv_0 = Conv2D(4, kernel_size, strides=(1, 1), padding='same', data_format='channels_last',  activation='relu')(Conv_t_do)
        Conv_0_do = Conv_0#tf.nn.dropout(Conv_0, keep_prob)
        #Conv_0_BN = BatchNormalization()(Conv_0)
        # Subpixel upsampling
        Conv_1 = tf.depth_to_space(Conv_0_do, 2)
        Conv_1_do = Conv_1#tf.nn.dropout(Conv_1, keep_prob)
        Conv_2 = Conv2D(1, kernel_size, strides=(1, 1), padding='same', data_format='channels_last',  activation='relu')(Conv_1_do)
        Conv_3 = Conv_2
    else:
        in_node_lstm = tf.expand_dims(tf.transpose(x_image,[0,3,1,2]),-1)
        
        # Fully-connected Keras layer
        Conv_0 = tf.keras.layers.ConvLSTM2D(hidden_num, kernel_size, name='state_layer1', padding= 'same', data_format='channels_last', return_sequences=False)(in_node_lstm)
        #convlstm_layer_bn_1 = tf.keras.layers.BatchNormalization(name='state_batch_norm1')(convlstm_layer_1)
        Conv_1 = tf.keras.layers.Conv2D(hidden_num, kernel_size, strides=(1, 1), padding='same', data_format='channels_last',  activation='relu')(Conv_0)
        Conv_2 = tf.depth_to_space(Conv_1, 2)
        Conv_3 = tf.keras.layers.Conv2D(1, kernel_size, strides=(1, 1), padding='same', data_format='channels_last',  activation='relu')(Conv_2)
        
        # output_std = tf.expand_dims(tf.math.reduce_std(x,axis=-1),axis=-1)
        # output_mean = tf.expand_dims(tf.math.reduce_mean(x,axis=-1),axis=-1)
        # output_end = convlstm_layer_conv_1+output_std
        # output_raw = tf.identity(convlstm_layer_conv_sub_1, 'output_raw') # just to preserve the name
    
        # psf_size = 9
        # psf_sigma = 2
        # psf_heatmap = util.matlab_style_gauss2D(shape = (psf_size,psf_size),sigma=psf_sigma)
        # gfilter = tf.reshape(psf_heatmap, [psf_size, psf_size, 1, 1])
        # output_end_psf = tf.nn.conv2d(output_raw, gfilter, [1, 1, 1, 1], padding="SAME")
        # print('We add an additional convolution to the output map!')


    # output:
    if(1):
        # no residual 
        output_end = Conv_3
        output_raw = tf.identity(output_end, 'output_raw') # just to preserve the name
    else:
        # residual for faster learning?
        output_std = tf.expand_dims(tf.math.reduce_std(x,axis=-1),axis=-1)
        output_mean = tf.expand_dims(tf.math.reduce_mean(x,axis=-1),axis=-1)

        output_end = Conv_2+output_mean
        output_raw = tf.identity(output_end, 'output_raw') # just to preserve the name

    psf_size = 9
    psf_sigma = 2
    psf_heatmap = util.matlab_style_gauss2D(shape = (psf_size,psf_size),sigma=psf_sigma)
    gfilter = tf.reshape(psf_heatmap, [psf_size, psf_size, 1, 1])
    output_end_psf = tf.nn.conv2d(output_raw, gfilter, [1, 1, 1, 1], padding="SAME")
    print('We add an additional convolution to the output map!')
    

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