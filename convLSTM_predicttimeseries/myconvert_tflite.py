import sofilstm.sofi as sofi
import sofilstm.train as train
from sofilstm import image_util

import numpy as np
import os
import matplotlib as mpl
import tensorflow as tf

mpl.rc('figure',  figsize=(12, 9))
mpl.rc('image', cmap='gray')

####################################################
####             PREPARE WORKSPACE               ###
####################################################
# here indicating the GPU you want to use. if you don't have GPU, just leave it.
gpu_ind = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ind; # 0,1,2,3

# Because we have real & imaginary part of our input, data_channels is set to 2
data_channels = 1
truth_channels = 1
features_root = 1

# args for training
batch_size = 1
ntimesteps = 30    # number of time-steps for one 3D volume
valid_size = batch_size  # batch size for validating (must be same as batch_size!)
optimizer = "adam"  # optimizer we want to use, 'adam' or 'momentum'
Nx, Ny = 128, 128 # final size of the image


# here specify the path of the model you want to load
model_path = './final/'



####################################################
####                lOAD MODEL                   ###
####################################################


tf.reset_default_graph()
net = sofi.SOFI(batchsize=batch_size, Nx=Nx, Ny=Ny, img_channels=1, features_root=features_root, ntimesteps=ntimesteps, cost="mean_squared_error")

net.saveTFLITE(model_path+'model.cpkt', outputmodelpath='converted_model.tflite')

net.simple_save(model_path+'model.cpkt', outputmodelpath='converted_model.pb')
