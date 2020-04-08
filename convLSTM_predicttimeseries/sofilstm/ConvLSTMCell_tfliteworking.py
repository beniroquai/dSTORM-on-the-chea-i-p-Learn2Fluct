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

#https://github.com/iwyoo/ConvLSTMCell-tensorflow/blob/master/ConvLSTMCell.py 
class ConvLSTMCell(object):
  """ Convolutional LSTM network cell (ConvLSTMCell).
  The implementation is based on http://arxiv.org/abs/1506.04214. 
   and `BasicLSTMCell` in TensorFlow. 
  """
  def __init__(self, hidden_num, filter_size=[3,3], 
               forget_bias=1.0, activation=tanh, name="ConvLSTMCell"):
    self.hidden_num = hidden_num
    self.filter_size = filter_size
    self.forget_bias = forget_bias
    self.activation = activation
    self.name = name

  def zero_state(self, batch_size, height, width):
    return tf.zeros([batch_size, height, width, self.hidden_num*2])

  def __call__(self, inputs, state, scope=None):
    """Convolutional Long short-term memory cell (ConvLSTM)."""
    with vs.variable_scope(scope or self.name): # "ConvLSTMCell"
      c, h = tf.split(state, 2, 3)

      # batch_size * height * width * channel
      concat = _conv([inputs, h], 4 * self.hidden_num, self.filter_size)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = tf.split(concat, 4, 3)

      new_c = (c * sigmoid(f + self.forget_bias) + sigmoid(i) *
               self.activation(j))
      new_h = self.activation(new_c) * sigmoid(o)
      new_state = tf.concat([new_c, new_h], 3)

      return new_h, new_state
      
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


tf.reset_default_graph()

height = 32
width = 32 
Ntime = 4
channel = 2
hidden_num = 1
batch_size = 1


if(0):
    input_ = tf.placeholder(tf.float32, [height, width, Ntime], 'input_')
    p_input = tf.reshape(input_, [batch_size, height, width, Ntime, channel])
else:
    input_ = tf.placeholder(tf.float32,[batch_size, height, width, Ntime, channel], 'input_')
    p_input = input_
mygroundtruth = tf.constant(np.zeros((height, width)))

p_input_list = tf.split(p_input,Ntime,3)
p_input_list = [tf.squeeze(p_input_, [3]) for p_input_ in p_input_list]

cell = ConvLSTMCell(hidden_num)
state = tf.truncated_normal(shape=[batch_size, width, height, hidden_num*2], stdv=0.1)
#state = cell.zero_state(batch_size, height, width)

with tf.variable_scope("ConvLSTM") as scope: # as BasicLSTMCell
    for i, p_input_ in enumerate(p_input_list):
        print('Concat: '+str(i))
        if i > 0: 
            scope.reuse_variables()
        # ConvCell takes Tensor with size [batch_size, height, width, channel].
        t_output, state = cell(p_input_, state)
        
output_ = tf.identity(tf.squeeze(t_output), name='output_')
myopt = tf.reduce_mean(input_tensor=tf.compat.v1.losses.mean_squared_error(output_, mygroundtruth))
sess = tf.compat.v1.Session()

init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(init)


converter = tf.lite.TFLiteConverter.from_session(sess, [input_], [output_])
#converter.experimental_new_converter = True  # Add this line
log_dir = './'
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open(log_dir+'mytflite.tflite', "wb").write(tflite_model)


