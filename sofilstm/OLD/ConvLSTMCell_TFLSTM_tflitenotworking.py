import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import init_ops
import os
from tensorflow.python.util import nest
import collections
import numpy as np 

# https://github.com/carlthome/tensorflow-convlstm-cell
import tensorflow as tf

class ConvLSTMCell(tf.nn.rnn_cell.RNNCell):
  """A LSTM cell with convolutions instead of multiplications.
  Reference:
    Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
  """

  def __init__(self, shape, filters, kernel, forget_bias=1.0, activation=tf.tanh, normalize=True,
               peephole=True, data_format='channels_last', reuse=None,scope=None):
    super(ConvLSTMCell, self).__init__(_reuse=reuse)
    self._kernel = kernel
    self._filters = filters
    self._forget_bias = forget_bias
    self._activation = activation
    self._normalize = normalize
    self._peephole = peephole
    self._scope=scope
    if data_format == 'channels_last':
        self._size = tf.TensorShape(shape + [self._filters])
        self._feature_axis = self._size.ndims
        self._data_format = None
    elif data_format == 'channels_first':
        self._size = tf.TensorShape([self._filters] + shape)
        self._feature_axis = 0
        self._data_format = 'NC'
    else:
        raise ValueError('Unknown data_format')

  @property
  def state_size(self):
    return tf.nn.rnn_cell.LSTMStateTuple(self._size, self._size)

  @property
  def output_size(self):
    return self._size

  # def zero_state(self, batch_size, height, width):
  #   return tf.zeros([batch_size, height, width,  self._filters,2])

  def call(self, x, state):
    with tf.variable_scope(self._scope,reuse=self._reuse):
      c, h = state
      x = tf.concat([x, h], axis=self._feature_axis)
      n = x.shape[-1].value
      m = 4 * self._filters if self._filters > 1 else 4
      W = tf.get_variable('kernel', self._kernel + [n, m])
      y = tf.nn.convolution(x, W, 'SAME', data_format=self._data_format)
      if not self._normalize:
        y += tf.get_variable('bias', [m], initializer=tf.zeros_initializer())
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

      state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

      return h, state

tf.reset_default_graph()

height = 32
width = 32 
Ntime = 4
channel = 1
hidden_num = 1
batch_size = 1
kernelsize = 3

# Create a placeholder for videos.
input_ = tf.placeholder(tf.float32, (batch_size, Ntime, height, width, channel))

# Add the ConvLSTM step.
cell = ConvLSTMCell([height, width], hidden_num, [kernelsize, kernelsize])
outputs, state = tf.nn.dynamic_rnn(cell, input_, dtype=input_.dtype)

output_ = tf.identity(tf.squeeze(outputs[:,-1,:,:,:]), name='output_')

mygroundtruth = tf.constant(np.zeros((height, width)))

myopt = tf.reduce_mean(input_tensor=tf.compat.v1.losses.mean_squared_error(output_, mygroundtruth))
sess = tf.compat.v1.Session()

init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(init)


converter = tf.lite.TFLiteConverter.from_session(sess, [input_], [output_])
converter.experimental_new_converter = True  # Add this line
log_dir = './'
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open(log_dir+'mytflite.tflite', "wb").write(tflite_model)


