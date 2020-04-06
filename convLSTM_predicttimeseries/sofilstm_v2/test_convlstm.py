#!pip install tensorflow==1.15.2
#!pip install tensorflow==2.2.0rc2
# Note this needs to happen before import tensorflow.
import os
os.environ['TF_ENABLE_CONTROL_FLOW_V2'] = '1'

import tensorflow as tf
tf.__version__




import numpy as np
import os


os.environ['TF_ENABLE_CONTROL_FLOW_V2'] = '1'

import convlstmcell as convlstmcell
tf.compat.v1.disable_eager_execution()
# Note this needs to happen before import tensorflow.
tf.compat.v1.reset_default_graph()
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
outputs, state = tf.compat.v1.lite.experimental.nn.dynamic_rnn(cell, input_, dtype=input_.dtype)
#outputs, state = tf.compat.v1.nn.static_rnn(cell, inputs, dtype=inputs.dtype)


output_ = tf.identity(outputs[:,-1,:,:,:], 'outputnode')

myopt = tf.reduce_mean(input_tensor=tf.compat.v1.losses.mean_squared_error(output_, mygroundtruth))
sess = tf.compat.v1.Session()
tf.compat.v1.summary.FileWriter(logdir='.\\log',graph=sess.graph)

init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(init)
log_dir = './'
tf.compat.v1.saved_model.simple_save(sess,
                        os.path.join(log_dir, "serve"),
                        inputs={'input': input_},
                        outputs={'output': output_})



if(0):
    converter = tf.lite.TFLiteConverter.from_session(sess, [input_], [output_])
    converter.experimental_new_converter = True  # Add this line
    
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()

else:

    import os
    import tensorflow as tf
    tf.__version__

    tf.compat.v1.enable_eager_execution()
    # alternative
    input_name = 'inputnode' # [inputs],
    output_name = 'outputnode' # [myoutput],
    
    log_dir = './'

   
    # tf.io.write_graph(sess.graph_def, log_dir, 'har.pbtxt') 
    #converter = tf.lite.TFLiteConverter.from_saved_model(sess, [input_], [output_])
    converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(log_dir, "serve"))
    #converter.experimental_new_converter = True  # Add this line
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.post_training_quantize = False
    tflite_model = converter.convert()
    open(log_dir+'mytflite.tflite', "wb").write(tflite_model)


    # saver.save(sess, save_path = "/tmp/checkpoint/har.ckpt")
    # saver = tf.train.Saver()
    #sess.close()

    
