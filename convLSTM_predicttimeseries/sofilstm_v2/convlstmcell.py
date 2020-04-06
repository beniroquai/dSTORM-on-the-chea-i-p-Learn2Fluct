#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 09:24:19 2020

@author: bene
"""


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
    
    y = tf.nn.convolution(input=x, filters=W, padding='SAME', data_format=self._data_format)
    
    if not self._normalize:
      y += tf.Variable(b_init, dtype=tf.float32, name='bias')
    j, i, f, o = tf.split(y, 4, axis=self._feature_axis)

    if self._peephole:
      i += tf.compat.v1.get_variable('W_ci', c.shape[1:]) * c
      f += tf.compat.v1.get_variable('W_cf', c.shape[1:]) * c

    if self._normalize:
      j = tf.contrib.layers.layer_norm(j)
      i = tf.contrib.layers.layer_norm(i)
      f = tf.contrib.layers.layer_norm(f)

    f = tf.sigmoid(f + self._forget_bias)
    i = tf.sigmoid(i)
    c = c * f + i * self._activation(j)

    if self._peephole:
      o += tf.compat.v1.get_variable('W_co', c.shape[1:]) * c

    if self._normalize:
      o = tf.contrib.layers.layer_norm(o)
      c = tf.contrib.layers.layer_norm(c)

    o = tf.sigmoid(o)
    h = o * self._activation(c)

    state = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c, h)

    return h, state

