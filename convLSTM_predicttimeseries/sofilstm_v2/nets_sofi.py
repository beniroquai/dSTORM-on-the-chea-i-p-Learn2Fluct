from __future__ import print_function, division, absolute_import, unicode_literals
import logging

import tensorflow as tf
import keras

import sofilstm.layers as layers
import sofilstm.util as util
import sofilstm.convlstm as convlstm 
import sofilstm.convlstmcell as convlstmExperimental


def sofi_decoder(x, y, keep_prob, phase, img_channels, truth_channels, features_root=16, kernel_size=3, pool_size=2, summaries=True):
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
    mysize = x.get_shape().as_list()
    batch_size = mysize[0]
    Nx = mysize[1]
    Ny = mysize[2]
    Ntime = mysize[3]
    x_image = tf.identity(x, 'x_input')
    #in_node = conv2d_bn_relu(x_image, kernel_size, features_root, keep_prob, phase, 'conv2feature_roots')

    if(0):
        in_node_lstm = tf.expand_dims(tf.transpose(a=x_image,perm=[0,3,1,2]),-1)
        cell = convlstm.ConvLSTMCell([Nx, Ny], features_root, [kernel_size, kernel_size])
        output, state = tf.compat.v1.nn.dynamic_rnn(cell, in_node_lstm, dtype=in_node_lstm.dtype)
        # make the order of the layers following the convention of the LSTM
        # https://github.com/leena201818/radioml/blob/master/experiment/lstm/ConvLSTM.py
        lstm_out = tf.transpose(a=output,perm=[0,2,3,4,2])[:,:,:,:,-1] # select the last time step as the one we want to use
    
        # Output Map
        if(1):
            print('Attention: We use sigmoid as last activation!')
            with tf.compat.v1.variable_scope("conv2d_1by1"):
                #output = conv2d(in_node, 1, truth_channels, keep_prob, 'conv2truth_channels')
                output_conv = layers.conv2d_bn_relu(lstm_out, 1, 1, keep_prob, phase, 'conv2truth_channels')
                output_std = tf.expand_dims(tf.math.reduce_std(x,axis=-1),axis=-1)
                output_mean = tf.expand_dims(tf.math.reduce_mean(input_tensor=x,axis=-1),axis=-1)
                output_end = output_conv+output_mean
                #output = conv2d_sigmoid(in_node, 1, truth_channels, keep_prob, 'conv2truth_channels')
                #output = deconv2d_bn_relu_res(in_node, 1, truth_channels, 1, keep_prob, phase, 'conv2truth_channels')
        else:
            output_end=lstm_out
            
        # add an additional convolution with a gaussian with the hope to reduce the extent of the structures
        psf_size = 9
        psf_sigma = 4
        psf_heatmap = util.matlab_style_gauss2D(shape = (psf_size,psf_size),sigma=psf_sigma)
        gfilter = tf.reshape(psf_heatmap, [psf_size, psf_size, 1, 1])
        output_end = tf.nn.conv2d(input=output_end, filters=gfilter, strides=[1, 1, 1, 1], padding="VALID")
        print('We add an additional convolution to the output map!')
    
    elif(0):
        n_features_convlstm = 1
        #in_node = tf.reshape(x_image, [batch_size, ntimesteps, insize_convlstm_m, insize_convlstm_n, n_features_convlstm]) # shape: [nbatch, ntime, nx, ny, nf]
        #lstm_input = tf.transpose(in_node, [1, 0, 2, 3, 4])  # to fit the time_major
        #convlstm(inputs=x_image, name = "conv_lstm_cell", Nx=8, Ny=8, nchannels=1024, n_hiddens = 32, kernel_shape=[3, 3])
                        
        # make the order of the layers following the convention of the LSTM
        in_node_lstm = tf.transpose(a=tf.expand_dims(x_image,-1),perm=[3,0,1,2,4])
        
        # Apply the convLSTM layer for all time-steps
        cell = convlstm.BasicConvLSTMCell([Nx,Ny], features_root, [kernel_size,kernel_size])
        output, state = tf.compat.v1.nn.dynamic_rnn(cell, in_node_lstm, dtype=in_node_lstm.dtype, time_major=True)
        
        # make the order of the layers following the convention of the LSTM
        lstm_out = tf.transpose(a=output,perm=[1,2,3,0,4])
        in_node = lstm_out[:,:,:,-1,:] # select the last time step as the one we want to use
  
        # Output Map
        print('Attention: We use sigmoid as last activation!')
        with tf.compat.v1.variable_scope("conv2d_1by1"):
            output_end = layers.conv2d_bn_relu_(in_node, kernel_size, 1, keep_prob, phase, 'conv2truth_channels')
    
    elif(0):
        in_node_lstm = tf.expand_dims(tf.transpose(a=x_image,perm=[0,3,1,2]),-1)
        
        # Fully-connected Keras layer
        convlstm_layer_1 = tf.keras.layers.ConvLSTM2D(features_root, kernel_size, name='state_layer1', padding= 'same', data_format='channels_last', return_sequences=False)(in_node_lstm)
        convlstm_layer_bn_1 = tf.keras.layers.BatchNormalization(name='state_batch_norm1')(convlstm_layer_1)
        convlstm_layer_conv_1 = tf.keras.layers.Conv2D(1, kernel_size, strides=(1, 1), padding='same', data_format='channels_last',  activation='relu')(convlstm_layer_bn_1)
        output_std = tf.expand_dims(tf.math.reduce_std(x,axis=-1),axis=-1)
        output_mean = tf.expand_dims(tf.math.reduce_mean(input_tensor=x,axis=-1),axis=-1)
        output_end = convlstm_layer_conv_1#+output_mean
        output_raw = tf.identity(output_end, 'output_raw') # just to preserve the name
    
        psf_size = 9
        psf_sigma = 2
        psf_heatmap = util.matlab_style_gauss2D(shape = (psf_size,psf_size),sigma=psf_sigma)
        gfilter = tf.reshape(psf_heatmap, [psf_size, psf_size, 1, 1])
        output_end_psf = tf.nn.conv2d(input=output_raw, filters=gfilter, strides=[1, 1, 1, 1], padding="SAME")
        print('We add an additional convolution to the output map!')


    elif(1):
        in_node_lstm = tf.transpose(a=x_image,perm=[0,3,1,2])
        in_node_lstm = tf.reshape(in_node_lstm, [batch_size, Ntime, Nx*Ny])
        shape = [Nx, Ny]
        
        # Add the ConvLSTM step.
        cell = convlstmExperimental.ConvLSTMCell_lite([Nx, Ny], features_root, [3, 3], timesteps=Ntime, normalize=False, peephole=False)
        outputs, state = tf.compat.v1.lite.experimental.nn.dynamic_rnn(cell, in_node_lstm, dtype=in_node_lstm.dtype)
        lstm_out = outputs[:,-1,:,:,:]
        
        
        convlstm_layer_conv_1 = tf.keras.layers.Conv2D(1, kernel_size, strides=(1, 1), padding='same', data_format='channels_last',  activation='relu')(lstm_out)
        output_std = tf.expand_dims(tf.math.reduce_std(x,axis=-1),axis=-1)
        output_mean = tf.expand_dims(tf.math.reduce_mean(input_tensor=x,axis=-1),axis=-1)
        output_end = convlstm_layer_conv_1#+output_mean
        output_raw = tf.identity(output_end, 'output_raw') # just to preserve the name
    
        psf_size = 9
        psf_sigma = 2
        psf_heatmap = util.matlab_style_gauss2D(shape = (psf_size,psf_size),sigma=psf_sigma)
        gfilter = tf.reshape(psf_heatmap, [psf_size, psf_size, 1, 1])
        output_end_psf = tf.nn.conv2d(input=output_raw, filters=gfilter, strides=[1, 1, 1, 1], padding="SAME")
        print('We add an additional convolution to the output map!')




    #if True: #summaries
    # Will reccord the summary for all images
    logging.info("Record all Image-Summaries")
    tf.compat.v1.summary.image('output_image', get_image_summary(output_end))
    tf.compat.v1.summary.image('output_image_psf', get_image_summary(output_end_psf))
    tf.compat.v1.summary.image('input_image', get_image_summary(tf.expand_dims(x[0,:,:,:],0)))
    tf.compat.v1.summary.image('groundtruth_image', get_image_summary(tf.expand_dims(y[0,:,:,:],0)))
    tf.compat.v1.summary.image('input_image_mean', get_image_summary(tf.expand_dims(tf.expand_dims(tf.math.reduce_mean(input_tensor=x[0,:,:,:],axis=-1),axis=0),axis=-1),0))
    tf.compat.v1.summary.image('input_image_std', get_image_summary(tf.expand_dims(tf.expand_dims(tf.math.reduce_std(x[0,:,:,:],axis=-1),axis=0),axis=-1),0))
       
    return output_end

def get_image_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """
    
    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(input_tensor=V)
    V /= tf.reduce_max(input_tensor=V)
    V *= 255
    
    img_w = tf.shape(input=img)[1]
    img_h = tf.shape(input=img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    V = tf.transpose(a=V, perm=(2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V