#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 09:24:19 2020

@author: bene
"""
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import init_ops
import os
from tensorflow.python.util import nest
import collections
import numpy as np 

import tensorflow as tf
import numpy as np

class ConvLSTMCell_lite(tf.compat.v1.lite.experimental.nn.TFLiteLSTMCell): #tf.compat.v1.nn.rnn_cell.RNNCell
  """A LSTM cell with convolutions instead of multiplications.
  Reference:
    Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
  """

  def __init__(self, shape, filters, kernel, timesteps = 1, forget_bias=1.0, activation=tf.tanh, normalize=True, peephole=True, reuse=None):
    super(ConvLSTMCell_lite, self).__init__(num_units=1)
    self._kernel = kernel
    self._filters = filters
    self._forget_bias = forget_bias
    self._activation = activation
    self._normalize = normalize
    self._peephole = peephole
    self._timesteps = timesteps
    
    self._shape = shape
    self._size = tf.TensorShape(shape + [self._filters])
    self._feature_axis = self._size.ndims
    self._data_format = None

  @property
  def state_size(self):
    return tf.compat.v1.nn.rnn_cell.LSTMStateTuple(self._size, self._size)

  @property
  def output_size(self):
    return self._size

  def call(self, x, state):
    c, h = state

    
    x = tf.reshape(x, [self._timesteps, self._shape[0], self._shape[1]])
    x = tf.concat([tf.expand_dims(x, -1), h], axis=self._feature_axis)
    
    #x = tf.concat([x, h], axis=self._feature_axis)
    n = self._feature_axis + self._timesteps
    m = 4 * self._filters if self._filters > 1 else 4

    W_init = np.random.rand(self._kernel[0], self._kernel[1], n, m)
    b_init = np.zeros((m))
    W = tf.Variable(W_init, dtype=tf.float32, name='kernel')
    
    y = tf.nn.convolution(x, W, padding='SAME', data_format=self._data_format)
    
    if not self._normalize:
      y += tf.Variable(b_init, dtype=tf.float32, name='bias')
    j, i, f, o = tf.split(y, 4, axis=self._feature_axis)

    if self._peephole:
      i += tf.get_variable('W_ci', c.shape[1:]) * c
      f += tf.get_variable('W_cf', c.shape[1:]) * c

    if self._normalize:
      j = tf.contrib.layers.layer_norm(j)
      i = tf.contrib.layers.layer_norm(i)
      f = tf.contrib.layers.layer_norm(f)

    f = tf.sigmoid(f + self._forget_bias)
    i = tf.sigmoid(i)
    c = c * f + i * self._activation(j)

    if self._peephole:
      o += tf.get_variable('W_co', c.shape[1:]) * c

    if self._normalize:
      o = tf.contrib.layers.layer_norm(o)
      c = tf.contrib.layers.layer_norm(c)

    o = tf.sigmoid(o)
    h = o * self._activation(c)

    state = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c, h)

    return h, state


# class ConvLSTMCell_orig(tf.nn.rnn_cell.RNNCell):
#   """A LSTM cell with convolutions instead of multiplications.
#   Reference:
#     Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
#   """

#   def __init__(self, shape, filters, kernel, forget_bias=1.0, activation=tf.tanh, normalize=True, peephole=True, data_format='channels_last', reuse=None):
#     super(ConvLSTMCell_orig, self).__init__(_reuse=reuse)
#     self._kernel = kernel
#     self._filters = filters
#     self._forget_bias = forget_bias
#     self._activation = activation
#     self._normalize = normalize
#     self._peephole = peephole
#     if data_format == 'channels_last':
#         self._size = tf.TensorShape(shape + [self._filters])
#         self._feature_axis = self._size.ndims
#         self._data_format = None
#     elif data_format == 'channels_first':
#         self._size = tf.TensorShape([self._filters] + shape)
#         self._feature_axis = 0
#         self._data_format = 'NC'
#     else:
#         raise ValueError('Unknown data_format')

#   @property
#   def state_size(self):
#     return tf.nn.rnn_cell.LSTMStateTuple(self._size, self._size)

#   @property
#   def output_size(self):
#     return self._size

#   def call(self, x, state):
#     c, h = state

#     x = tf.concat([x, h], axis=self._feature_axis)
#     n = x.shape[-1].value
#     m = 4 * self._filters if self._filters > 1 else 4
#     W = tf.get_variable('kernel', self._kernel + [n, m])
#     y = tf.nn.convolution(x, W, padding='SAME', data_format=self._data_format)
#     if not self._normalize:
#       y += tf.get_variable('bias', [m], initializer=tf.zeros_initializer())
#     j, i, f, o = tf.split(y, 4, axis=self._feature_axis)

#     if self._peephole:
#       i += tf.get_variable('W_ci', c.shape[1:]) * c
#       f += tf.get_variable('W_cf', c.shape[1:]) * c

#     if self._normalize:
#       j = tf.contrib.layers.layer_norm(j)
#       i = tf.contrib.layers.layer_norm(i)
#       f = tf.contrib.layers.layer_norm(f)

#     f = tf.sigmoid(f + self._forget_bias)
#     i = tf.sigmoid(i)
#     c = c * f + i * self._activation(j)

#     if self._peephole:
#       o += tf.get_variable('W_co', c.shape[1:]) * c

#     if self._normalize:
#       o = tf.contrib.layers.layer_norm(o)
#       c = tf.contrib.layers.layer_norm(c)

#     o = tf.sigmoid(o)
#     h = o * self._activation(c)

#     state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

#     return h, state


# class ConvLSTMCell(tf.compat.v1.nn.rnn_cell.RNNCell): #
#   """A LSTM cell with convolutions instead of multiplications.
#   Reference:
#     Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
#   """

#   def __init__(self, shape, filters, kernel, timesteps = 1, forget_bias=1.0, activation=tf.tanh, normalize=True, peephole=True, reuse=None):
#     super(ConvLSTMCell, self).__init__(_reuse=reuse)
#     self._kernel = kernel
#     self._filters = filters
#     self._forget_bias = forget_bias
#     self._activation = activation
#     self._normalize = normalize
#     self._peephole = peephole
    
#     self._size = tf.TensorShape(shape + [self._filters])
#     self._feature_axis = self._size.ndims
#     self._data_format = None

#   @property
#   def state_size(self):
#     return tf.compat.v1.nn.rnn_cell.LSTMStateTuple(self._size, self._size)

#   @property
#   def output_size(self):
#     return self._size

#   def call(self, x, state):
#     c, h = state

#     x = tf.concat([tf.expand_dims(x, -1), h], axis=self._feature_axis)
#     #x = tf.concat([x, h], axis=self._feature_axis)
#     n = x.shape[-1].value
#     m = 4 * self._filters if self._filters > 1 else 4

#     W_init = np.random.rand(self._kernel[0], self._kernel[1], n, m)
#     b_init = np.zeros((m))
#     W = tf.Variable(W_init, dtype=tf.float32, name='kernel')
    
#     y = tf.nn.convolution(x, W, padding='SAME', data_format=self._data_format)
    
#     if not self._normalize:
#       y += tf.Variable(b_init, dtype=tf.float32, name='bias')
#     j, i, f, o = tf.split(y, 4, axis=self._feature_axis)

#     if self._peephole:
#       i += tf.get_variable('W_ci', c.shape[1:]) * c
#       f += tf.get_variable('W_cf', c.shape[1:]) * c

#     if self._normalize:
#       j = tf.contrib.layers.layer_norm(j)
#       i = tf.contrib.layers.layer_norm(i)
#       f = tf.contrib.layers.layer_norm(f)

#     f = tf.sigmoid(f + self._forget_bias)
#     i = tf.sigmoid(i)
#     c = c * f + i * self._activation(j)

#     if self._peephole:
#       o += tf.get_variable('W_co', c.shape[1:]) * c

#     if self._normalize:
#       o = tf.contrib.layers.layer_norm(o)
#       c = tf.contrib.layers.layer_norm(c)

#     o = tf.sigmoid(o)
#     h = o * self._activation(c)

#     state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

#     return h, state



#https://github.com/iwyoo/ConvLSTMCell-tensorflow/blob/master/ConvLSTMCell.py 
class ConvLSTMCell(object):
    """ Convolutional LSTM network cell (ConvLSTMCell).
    The implementation is based on http://arxiv.org/abs/1506.04214. 
     and `BasicLSTMCell` in TensorFlow. 
    """
    def __init__(self, num_features, filter_size=[3,3], 
                 forget_bias=1.0, activation=tanh, name="ConvLSTMCell"):
      self.num_features = num_features
      self.filter_size = filter_size
      self.forget_bias = forget_bias
      self.activation = activation
      self.name = name

    def zero_state(self, batch_size, height, width):
        return tf.zeros([batch_size, height, width, self.num_features*2])

    def __call__(self, inputs, state, scope=None):
        """Convolutional Long short-term memory cell (ConvLSTM)."""
        with vs.variable_scope(scope or self.name): # "ConvLSTMCell"
            c, h = tf.split(state, 2, 3)

            # # batch_size * height * width * channel
            # concat = _conv([inputs, h], 4 * self.num_features, self.filter_size)
              
            # # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            # i, j, f, o = tf.split(concat, 4, 3)
              
            # new_c = (c * sigmoid(f + self.forget_bias) + sigmoid(i) *
            #          self.activation(j))
            # new_h = self.activation(new_c) * sigmoid(o)
            # new_state = tf.concat([new_c, new_h], 3)

              # return new_h, new_state
      
        self._normalize = False
        self._peephole = True
        self._forget_bias= 1.0

        inputs = tf.concat([inputs, h], axis=-1)#self._feature_axis)
        n = inputs.shape[-1].value
        m = 4 * self.num_features if self.num_features > 1 else 4
        W = tf.get_variable('kernel', self.filter_size + [n, m])
        y = tf.nn.convolution(inputs, W, 'SAME', data_format=None)
        if not self._normalize:
            y += tf.get_variable('bias', [m], initializer=tf.zeros_initializer())
        j, i, f, o = tf.split(y, 4, axis=-1)#self._feature_axis)
        
        if self._peephole:
            i += tf.get_variable('W_ci', c.shape[1:]) * c
            f += tf.get_variable('W_cf', c.shape[1:]) * c
              
        if self._normalize:
            j = tf.contrib.layers.layer_norm(j,reuse=True)
            i = tf.contrib.layers.layer_norm(i,reuse=True)
            f = tf.contrib.layers.layer_norm(f,reuse=True)
        
        f = tf.sigmoid(f + self._forget_bias)
        i = tf.sigmoid(i)
        c = c * f + i * self.activation(j)
        
        if self._peephole:
            o += tf.get_variable('W_co', c.shape[1:]) * c
        
        if self._normalize:
            o = tf.contrib.layers.layer_norm(o)
            c = tf.contrib.layers.layer_norm(c)
              
        o = tf.sigmoid(o)
        h = o * self.activation(c)
        
        state = tf.concat([c, h], 3) #tf.nn.rnn_cell.LSTMStateTuple(c, h)

        return h, state
      
def _conv(args, output_size, filter_size, stddev=0.001, bias=True, bias_start=0.0, scope=None):
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 3.
  # (batch_size x height x width x arg_size)
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  height = shapes[0][1]
  width  = shapes[0][2]
  for shape in shapes:
    if len(shape) != 4:
      raise ValueError("Conv is expecting 3D arguments: %s" % str(shapes))
    if not shape[3]:
      raise ValueError("Conv expects shape[3] of arguments: %s" % str(shapes))
    if shape[1] == height and shape[2] == width:
      total_arg_size += shape[3]
    else :
      raise ValueError("Inconsistent height and width size in arguments: %s" % str(shapes))
  
  with vs.variable_scope(scope or "Conv"):
    kernel = vs.get_variable("Kernel", 
      [filter_size[0], filter_size[1], total_arg_size, output_size],
      initializer=init_ops.truncated_normal_initializer(stddev=stddev))
    
    if len(args) == 1:
      res = tf.nn.conv2d(args[0], kernel, [1, 1, 1, 1], padding='SAME')
    else:
      res = tf.nn.conv2d(tf.concat(args, 3), kernel, [1, 1, 1, 1], padding='SAME')

    if not bias: return res
    bias_term = vs.get_variable( "Bias", [output_size],
      initializer=init_ops.constant_initializer(bias_start))
  return res + bias_term
