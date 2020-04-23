# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 08:55:36 2020

@author: diederichbenedict
"""

import tensorflow as tf
import convlstmbasic as basiclstm
import numpy as np

shape = [100,100]
batch_size = 1
timesteps = 10
filter_size = [3,3]
num_features = 4
myinputs = np.float32(np.random.rand(batch_size, timesteps, shape[0], shape[1], num_features))
inputs = tf.Variable(myinputs, name='testinput')
cell = basiclstm.BasicConvLSTMCell(shape, filter_size, num_features)
hidden = cell.zero_state(batch_size, tf.float32) 


y_1, hidden = cell(inputs, hidden)