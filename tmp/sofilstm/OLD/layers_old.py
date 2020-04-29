# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Modified on Mar, 2020 based on the work of jakeret

author: Benedict 
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import tensorflow as tf


def log(x, base):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(base, dtype=numerator.dtype))
  return numerator / denominator

def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def rescale(array_x): # convert to [0,1]
    amax = tf.reduce_max(array_x, axis=1, keep_dims=True)
    amin = tf.reduce_min(array_x, axis=1, keep_dims=True)
    rescaled = array_x - amin
    rescaled = rescaled / amax
    return rescaled

# receives an array of images and return the mse per image.
# size ~ num of pixels in the img
def mse_array(array_x, array_y, size):
    rescale_x = array_x
    rescale_y = array_y
    se = tf.reduce_sum(tf.squared_difference(rescale_x, rescale_y), 1)
    inv_size = tf.to_float(1/size)
    return tf.scalar_mul(inv_size, se)

def conv2d_bn_relu(x, w_size, num_outputs, keep_prob_, phase, scope): # output size should be the same.
    conv_2d = tf.contrib.layers.conv2d(x, num_outputs, w_size,
                                        activation_fn=tf.nn.relu,   # elu is an alternative
                                        normalizer_fn=tf.layers.batch_normalization,
                                        normalizer_params={'training': phase},
                                        scope=scope)

    return tf.nn.dropout(conv_2d, keep_prob_)

def deconv2d_bn_relu_res(x, w_size, num_outputs, stride, keep_prob_, phase, scope):
    # this is very likely responssible for the checkerboard
    # remove checkerboard artifact have a look at the distill paper"
    _b, h, w, _c = x.shape
    initializer = tf.random_normal_initializer(0, 0.02)
    resized_input = tf.image.resize_images(x, [h * 2, w * 2], method=tf.image.ResizeMethod.BILINEAR)
    conv_2d = tf.layers.separable_conv2d(resized_input, num_outputs,
                                        kernel_size=w_size,
                                        strides=stride,
                                        padding="same",
                                        depthwise_initializer=initializer,
                                        pointwise_initializer=initializer)\
                                        #, scope = scope)
    return tf.nn.dropout(conv_2d, keep_prob_)



def deconv2d_bn_relu(x, w_size, num_outputs, stride, keep_prob_, phase, scope):
    conv_2d = tf.contrib.layers.conv2d_transpose(x, num_outputs, w_size,
                                                stride=stride,
                                                activation_fn=tf.nn.relu,   # elu is an alternative
                                                normalizer_fn=tf.layers.batch_normalization,
                                                normalizer_params={'training': phase},
                                                scope=scope)

    return tf.nn.dropout(conv_2d, keep_prob_)

def conv2d_bn(x, w_size, num_outputs, keep_prob_, phase, scope):
    conv_2d = tf.contrib.layers.conv2d(x, num_outputs, w_size,
                                        activation_fn=None,
                                        normalizer_fn=tf.layers.batch_normalization,
                                        normalizer_params={'training': phase},
                                        scope=scope)
    return conv_2d

def conv2d(x, w_size, num_outputs, keep_prob_, scope):
    conv_2d = tf.contrib.layers.conv2d(x, num_outputs, w_size,
                                        activation_fn=None,
                                        normalizer_fn=None,
                                        scope=scope)
    return conv_2d

def conv2d_sigmoid(x, w_size, num_outputs, keep_prob_, scope):
    conv_2d = tf.contrib.layers.conv2d(x, num_outputs, w_size,
                                        activation_fn=tf.nn.sigmoid,
                                        normalizer_fn=None,
                                        scope=scope)
    return conv_2d

def max_pool(x,n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME')

def concat(x1,x2):
    return tf.concat([x1, x2], 3)   

def convlstm(inputs, name = "conv_lstm_cell", Nx=8, Ny=8, nchannels=1024, n_hiddens = 32, kernel_shape=[3, 3]):
    ''' Basic idea: Have a convlstm in the bottleneck layer, where each 
    timestep is fed into one hidden layer of the convLSTM. This should nodel the 
    sequential nature of the data along the propagation axis (i.e. "Timesteps")
    The model also outputs all convlstm cell's outputs which are concatenated 
    and decoded by the U_Net decoder. All other Skip connectsions won't be 
    treated this way.  Information transfer should be ensured, because LSTM is 
    hard to train anyway. '''
    #https://github.com/yz-cnsdqz/exercise2_VideoParsing/
    #https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/
    #https://github.com/Apm5/my_deep_learning_code/ <- Looks good!
    # Do we need to stack/unstack anything?
    
    convlstm_layer= tf.contrib.rnn.ConvLSTMCell(
                conv_ndims=2,
                input_shape=[Nx, Ny, nchannels],
                output_channels=n_hiddens,
                kernel_shape=kernel_shape,
                use_bias=True,
                skip_connection=False,
                forget_bias=1.0,
                initializers=None,
                name=name)
    
    mysize = tf.shape(inputs) 
    batch_size = mysize[0]
    initial_state = convlstm_layer.zero_state(batch_size, dtype=tf.float32)
    outputs,_=tf.nn.dynamic_rnn(convlstm_layer,inputs,initial_state=initial_state,time_major=False,dtype="float32")
    return outputs
