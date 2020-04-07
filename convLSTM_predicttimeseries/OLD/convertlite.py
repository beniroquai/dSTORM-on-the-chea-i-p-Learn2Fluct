# Need to do that for tflite and need to do it before importing tensorflow
import os
os.environ['TF_ENABLE_CONTROL_FLOW_V2'] = '1'


import sofilstm.sofi as sofi
import sofilstm.train as train
from sofilstm import image_util

import scipy.io as spio
import numpy as np
import os
import NanoImagingPack as nip
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
import tensorflow as tf 
mpl.rc('figure',  figsize=(12, 9))
mpl.rc('image', cmap='gray')

####################################################
####             PREPARE WORKSPACE               ###
####################################################


# Because we have real & imaginary part of our input, data_channels is set to 2
data_channels = 1
truth_channels = 1

# here indicating the GPU you want to use. if you don't have GPU, just leave it.
gpu_ind = '-1'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ind; # 0,1,2,3


# here specify the path of the model you want to load
model_path = './networks/upsamping_2_noconv_100x100_2_gpu0/models/final/'


####################################################
####                 FUNCTIONS                   ###
####################################################

# make the data a 4D vector
def preprocess(data, mysize=None):
    if ~(mysize is None):
        data = nip.extract(data, mysize)
    data -= np.min(data)
    data = data/np.max(data)
    return data

####################################################
####                lOAD MODEL                   ###
####################################################

# Because we have real & imaginary part of our input, data_channels is set to 2
data_channels = 1
truth_channels = 1
features_root = 16

# args for training
batch_size = 1  # batch size for training
ntimesteps = 1  # number of time-steps for one 3D volume
valid_size = batch_size  # batch size for validating (must be same as batch_size!)
Nx, Ny = 100, 100

path_mydata = './test/sofi.mat'
path_mydata = './test/sofi_ecoli.mat'

#preparing training data
from skimage.transform import resize
data_mat = spio.loadmat(path_mydata , squeeze_me=True)
upscaling = 4
data = np.float32(np.array(data_mat['sofi']))
Nx,Ny,Nz = data.shape[0],data.shape[1],data.shape[2]
data = resize(data, (Nx*upscaling, Nx*upscaling, Nz))
data = preprocess(data, mysize=(Nx*upscaling,Ny*upscaling,ntimesteps))
data = np.expand_dims(data,0)
#data = np.concatenate((data,data), 0)

tf.reset_default_graph()
net = sofi.SOFI(batchsize=batch_size, Nx=Nx, Ny=Ny, img_channels=1, features_root=features_root, ntimesteps=ntimesteps, cost="mean_squared_error")

# Initialize variables

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

saver = tf.train.Saver()
saver.restore(sess, model_path+'model.cpkt')
        
converter = tf.lite.TFLiteConverter.from_session(sess, [net.x], [net.recons])
converter.experimental_new_converter = True  # Add this line

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

open(outputmodelpath, "wb").write(tflite_model)


