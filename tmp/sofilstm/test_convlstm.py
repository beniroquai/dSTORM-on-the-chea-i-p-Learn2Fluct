#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 09:33:39 2020

@author: bene
"""
import os
os.environ['TF_ENABLE_CONTROL_FLOW_V2'] = '1'


import numpy as np
import tensorflow as tf
import convlstmcell as convlstmcell
# Note this needs to happen before import tensorflow.
tf.reset_default_graph()
batch_size = 1
timesteps = 10
shape = [100, 100]
kernel = [3, 3]
channels = 3
filters = 12

# Create a placeholder for videos.
input_ = tf.compat.v1.placeholder(tf.float32, (batch_size, timesteps, (shape[0]*shape[1])), 'inputnode')# + [channels])
mygroundtruth = tf.constant(np.zeros((1,100,100,12)), tf.float32)
#inputs = tf.compat.v1.placeholder(tf.float32, [batch_size, timesteps] + shape)# + [channels])

# Add the ConvLSTM step.
cell = convlstmcell.ConvLSTMCell_lite(shape, filters, kernel, timesteps=timesteps, normalize=False, peephole=False)
#outputs, state = tf.compat.v1.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype)
outputs, state = tf.lite.experimental.nn.dynamic_rnn(cell, input_, dtype=input_.dtype)
#outputs, state = tf.compat.v1.nn.static_rnn(cell, inputs, dtype=inputs.dtype)


output_ = tf.identity(outputs[:,-1,:,:,:], 'outputnode')

myopt = tf.reduce_mean(tf.losses.mean_squared_error(output_, mygroundtruth))
sess = tf.Session()
tf.compat.v1.summary.FileWriter(logdir='.\\log',graph=sess.graph)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
   
if(0):
    converter = tf.lite.TFLiteConverter.from_session(sess, [input_], [output_])
    converter.experimental_new_converter = True  # Add this line
    
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()

else:
    # alternative
    input_name = 'inputnode' # [inputs],
    output_name = 'outputnode' # [myoutput],
    
    log_dir = './tmp/checkpoint2'
    tf.train.write_graph(sess.graph_def, log_dir, 'har.pbtxt') 
    
    
    #converter = tf.lite.TFLiteConverter.from_saved_model(sess, [input_], [output_])
    
    tf.saved_model.simple_save(sess,
                                os.path.join(log_dir, "serve"),
                                inputs={'input': input_},
                                outputs={'output': output_})
    converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(log_dir, "serve"))
    # saver.save(sess, save_path = "/tmp/checkpoint/har.ckpt")
    # saver = tf.train.Saver()
    #sess.close()
    

    
    from tensorflow.python.tools import freeze_graph
    
    MODEL_NAME = 'har'
    
    input_graph_path = '/tmp/checkpoint/' + MODEL_NAME+'.pbtxt'
    checkpoint_path = '/tmp/checkpoint/' +MODEL_NAME+'.ckpt'
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_frozen_graph_name = 'frozen_'+MODEL_NAME+'.pb'
    
    
    freeze_graph.freeze_graph(input_graph_path, input_saver="",
                              input_binary=False, input_checkpoint=checkpoint_path, 
                              output_node_names=output_name, restore_op_name="save/restore_all",
                              filename_tensor_name="save/Const:0", 
                              output_graph=output_frozen_graph_name, clear_devices=True, initializer_nodes="")
    
    
