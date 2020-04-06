# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 18:00:33 2020

@author: diederichbenedict
"""

import tensorflow as tf

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph 
model_dir = './networks/upsamping_2_noconv_100x100_2_gpu0/models/final/'
output_node_names = 'output_raw'
input_node_name = 'x_input'


# Convert the model.
converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)
tflite_model = converter.convert()


# We retrieve our checkpoint fullpath
checkpoint = tf.train.get_checkpoint_state(model_dir)
input_checkpoint = checkpoint.model_checkpoint_path

#freeze_graph(model_dir, output_node_names)



# We precise the file fullname of our freezed graph
absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
output_graph = absolute_model_dir + "/saved_model.pb"

# We clear devices to allow TensorFlow to control on which device it will load operations
clear_devices = True

# We start a session using a temporary fresh Graph
with tf.Session(graph=tf.Graph()) as sess:
    # We import the meta graph in the current default Graph
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We restore the weights
    saver.restore(sess, input_checkpoint)

    # We use a built-in TF helper to export variables to constants
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, # The session is used to retrieve the weights
        tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
        output_node_names.split(",") # The output node names are used to select the usefull nodes
    ) 

    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))



##### Optimize graph	
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

inputGraph = tf.GraphDef()
with tf.gfile.Open(absolute_model_dir+'/'+'saved_model.pb', "rb") as f:
    data2read = f.read()
    inputGraph.ParseFromString(data2read)

outputGraph = optimize_for_inference_lib.optimize_for_inference(
                inputGraph,
                [input_node_name], # an array of the input node(s)
                [output_node_names], # an array of output nodes
                tf.int32.as_datatype_enum)

        # Save the optimized graph

f = tf.gfile.FastGFile(absolute_model_dir+'/'+"saved_model_opt.pb", "w")
f.write(outputGraph.SerializeToString())    

''' TFLITE STUFF'''



output_graph = "./tflitegraph.pb"
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.tools.freeze_graph import freeze_graph


# ckpt = tf.train.get_checkpoint_state(input_checkpoint)
# input_checkpoint = ckpt.model_checkpoint_path
# if not (ckpt and input_checkpoint):
#     return False
tf.reset_default_graph()
saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
# graph = tf.get_default_graph  # Get the default diagram
method = 1
if method == 0:
    sess = tf.Session()
    saver.restore(sess, input_checkpoint)    # Restore the graph and get the data
    output_graph_def = graph_util.convert_variables_to_constants(
        sess=sess,
        input_graph_def=sess.graph_def,
        output_node_names=output_node_names.split(',')
    )
    with tf.gfile.GFile("./save/pb/new_frozen_model.pb", "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))  # print current graph operation nodes
elif method == 1:
    sess = tf.Session()
    saver.restore(sess, input_checkpoint)  # Restore the graph and get the data
    converter = tf.lite.TFLiteConverter.from_session(sess, [img], [out])
    
    temp_pb = 'temp_model.pb'
    temp_frozen_pb = './save/pb/temp_model.pb'
    tf.train.write_graph(sess.graph_def, './save/pb', temp_pb)
    freeze_graph(input_graph='./save/pb/temp_model.pb',
                 input_saver='',
                 input_binary=False,
                 input_checkpoint=input_checkpoint,
                 output_node_names=output_node_names,
                 restore_op_name='save/restore_all',
                 filename_tensor_name='save/Const:0',
                 output_graph=output_graph,
                 clear_devices=True,
                 initializer_nodes='')
    print('*******************', output_graph)
output_graph = "./save/pb/new_frozen_model.pb"    
converter = tf.lite.TFLiteConverter.from_frozen_graph(output_graph, input_node_name, output_node_names)
tflite_model = converter.convert()
with open("./save/temp_model.tflite", "wb") as f:
    f.write(tflite_model)
    print("[Info]: Covert ckpt file to tflite is Ok!")




