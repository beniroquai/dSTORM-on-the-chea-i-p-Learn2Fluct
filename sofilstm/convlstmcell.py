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


#https://github.com/iwyoo/ConvLSTMCell-tensorflow/blob/master/ConvLSTMCell.py 
class ConvLSTMCell(object):
    """ Convolutional LSTM network cell (ConvLSTMCell).
    The implementation is based on http://arxiv.org/abs/1506.04214. 
     and `BasicLSTMCell` in TensorFlow. 
    """
    def __init__(self, num_features, filter_size=[3,3], 
                 forget_bias=1.0, activation=tanh, name="ConvLSTMCell",  
                 is_normalize=False, is_peephole=False):
      self.num_features = num_features
      self.filter_size = filter_size
      self.forget_bias = forget_bias
      self.activation = activation
      self.name = name
      self._normalize = is_normalize
      self._peephole = is_peephole
      self._forget_bias= 1.0

    def zero_state(self, batch_size, height, width):
        return tf.zeros([batch_size, height, width, self.num_features*2])

    def __call__(self, inputs, state, scope=None,):
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

        inputs = tf.concat([inputs, h], axis=-1)
        n = inputs.shape[-1].value
        m = 4 * self.num_features if self.num_features > 1 else 4
        
        init = tf.keras.initializers.glorot_uniform()
        W = tf.get_variable('kernel', self.filter_size + [n, m], initializer=init)
        y = tf.nn.convolution(inputs, W, 'SAME', data_format=None)
        if not self._normalize:
            y += tf.get_variable('bias', [m], initializer=tf.zeros_initializer())
        j, i, f, o = tf.split(y, 4, axis=-1)#self._feature_axis)
        
        if self._peephole:
            i += tf.get_variable('W_ci', c.shape[1:]) * c
            f += tf.get_variable('W_cf', c.shape[1:]) * c
              
        if self._normalize:
            f = tf.keras.layers.LayerNormalization()(f)
            i = tf.keras.layers.LayerNormalization()(i)
            j = tf.keras.layers.LayerNormalization()(j)            
        
        f = tf.sigmoid(f + self._forget_bias)
        i = tf.sigmoid(i)
        c = c * f + i * self.activation(j)
        
        if self._peephole:
            o += tf.get_variable('W_co', c.shape[1:]) * c
        
        if self._normalize:
            o = tf.keras.layers.LayerNormalization()(o)  
            c = tf.keras.layers.LayerNormalization()(c)            
              
        #o = tf.keras.activations.hard_sigmoid(o)
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
