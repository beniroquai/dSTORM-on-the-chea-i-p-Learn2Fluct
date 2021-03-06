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
    strides = 1
    cnn_weights = tf.get_variable("cnn_weights_bn_relu", shape=[3, 3, x.shape[-1], num_outputs], initializer=tf.initializers.glorot_uniform())
    cnn_bias = tf.get_variable("cnn_bias_bn_relu", shape=num_outputs, initializer=tf.initializers.glorot_uniform())    
    out = tf.nn.conv2d(x, cnn_weights, padding="SAME", strides = [1, strides, strides, 1])
    out = tf.nn.bias_add(out, cnn_bias)
    out =  tf.nn.relu(out)
    return tf.nn.dropout(out, keep_prob_)


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




def conv2d(x, w_size, num_outputs, keep_prob_, scope):
    strides = 1
    cnn_weights = tf.get_variable("cnn_weights_bn_relu", shape=[3, 3, x.shape[-1], num_outputs], initializer=tf.initializers.glorot_uniform())
    cnn_bias = tf.get_variable("cnn_bias_bn_relu", shape=num_outputs, initializer=tf.initializers.glorot_uniform())    
    out = tf.nn.conv2d(x, cnn_weights, padding="SAME", strides = [1, strides, strides, 1])
    return tf.nn.bias_add(out, cnn_bias)


def conv2d_sigmoid(x, w_size, num_outputs, keep_prob_, scope):
    strides = 1
    cnn_weights = tf.get_variable("cnn_weights_bn_relu", shape=[3, 3, x.shape[-1], num_outputs], initializer=tf.initializers.glorot_uniform())
    cnn_bias = tf.get_variable("cnn_bias_bn_relu", shape=num_outputs, initializer=tf.initializers.glorot_uniform())    
    out = tf.nn.conv2d(x, cnn_weights, padding="SAME", strides = [1, strides, strides, 1])
    out = tf.nn.bias_add(out, cnn_bias)
    return tf.nn.sigmoid(out)


def max_pool(x,n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME')

def concat(x1,x2):
    return tf.concat([x1, x2], 3)   
