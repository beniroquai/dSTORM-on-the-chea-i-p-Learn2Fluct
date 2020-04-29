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
Modified on Feb, 2018 based on the work of Apm5

author: bene
'''
# https://raw.githubusercontent.com/leena201818/radioml/master/experiment/lstm/ConvLSTM.py
import tensorflow as tf
import numpy as np

class BasicConvLSTMCell(tf.contrib.rnn.RNNCell):

  def __init__(self, shape, num_filters, kernel_size, forget_bias=1.0,
               input_size=None, state_is_tuple=True, activation=tf.nn.tanh, reuse=None):
    self._shape = shape
    self._num_filters = num_filters
    self._kernel_size = kernel_size
    self._size = tf.TensorShape(shape+[self._num_filters])

    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation
    self._reuse = reuse
  @property
  def state_size(self):
    return (tf.contrib.rnn.LSTMStateTuple(self._size, self._size)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._size

  def __call__(self, inputs, state, scope=None):
    # we suppose inputs to be [time, batch_size, row, col, channel]
    with tf.variable_scope(scope or "basic_convlstm_cell", reuse=self._reuse):
      if self._state_is_tuple:
        c, h = state
      else:
        c, h = array_ops.split(value=state, num_or_size_splits=2, axis=3)

      inp_channel = inputs.get_shape().as_list()[-1]+self._num_filters
      out_channel = self._num_filters * 4
      concat = tf.concat([inputs, h], axis=3)

      kernel = tf.get_variable('kernel', shape=self._kernel_size+[inp_channel, out_channel])
      concat = tf.nn.conv2d(concat, filter=kernel, strides=(1,1,1,1), padding='SAME')

      i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=3)

      new_c = (c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) *
               self._activation(j))
      new_h = self._activation(new_c) * tf.sigmoid(o)
      if self._state_is_tuple:
        new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
      else:
        new_state = tf.concat([new_c, new_h], 3)
      return new_h, new_state
  
# https://github.com/carlthome/tensorflow-convlstm-cell
class ConvLSTMCell(tf.nn.rnn_cell.RNNCell):
  """A LSTM cell with convolutions instead of multiplications.
  Reference:
    Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
  """

  def __init__(self, shape, filters, kernel, forget_bias=1.0, activation=tf.tanh, normalize=True, peephole=True, data_format='channels_last', reuse=None):
    super(ConvLSTMCell, self).__init__(_reuse=reuse)
    self._kernel = kernel
    self._filters = filters
    self._forget_bias = forget_bias
    self._activation = activation
    self._normalize = normalize
    self._peephole = peephole
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

  def call(self, x, state):
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

if __name__ == '__main__':
    if (0):
        Nx, Ny = 200,200
        ntimesteps = 50
        batchsize = 2
        nchanels = 1
        size_kernel = 3
        num_kernel = 6
        inputs=tf.placeholder(tf.float32, [ntimesteps,batchsize,Nx,Ny,nchanels])
        # we suppose inputs to be [time, batch_size, row, col, channel]
        # (self, shape, num_filters, kernel_size, forget_bias=1.0,
        #             input_size=None, state_is_tuple=True, activation=tf.nn.tanh, reuse=None)
        
        cell = BasicConvLSTMCell([Nx,Ny], num_kernel, [size_kernel,size_kernel])
        

        outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype, time_major=True)
        with tf.Session() as sess:
            inp = np.random.normal(size=(ntimesteps,batchsize,Nx,Ny,nchanels))
            sess.run(tf.global_variables_initializer())
            o, s = sess.run([outputs, state], feed_dict={inputs:inp})
            print (o.shape) #(5,2,3,3,6)'
    else:
            
        
        batch_size = 32
        timesteps = 100
        shape = [640, 480]
        kernel = [3, 3]
        channels = 1
        filters = 12
        # Create a placeholder for videos.
        inputs = tf.placeholder(tf.float32, [batch_size, timesteps] + shape + [channels])
        
        cell = ConvLSTMCell(shape, filters, kernel)
        outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype)
        
                
                
