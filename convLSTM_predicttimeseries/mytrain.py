# Need to do that for tflite and need to do it before importing tensorflow
import os
os.environ['TF_ENABLE_CONTROL_FLOW_V2'] = '1'

import sofilstm.sofi as sofi
import sofilstm.train as train
from sofilstm import image_util

import scipy.io as spio
import numpy as np
import os
from sys import platform
# This is taken from the ScaDec from Ulugbek Kamilov 2018 et al. which is a copy of the tf_unet



####################################################
####              HYPER-PARAMETERS               ###
####################################################

# Because we have real & imaginary part of our input, data_channels is set to 2
data_channels = 1
truth_channels = 1
features_root = 1
# args for training
batch_size = 4     # batch size for training
ntimesteps = 30    # number of time-steps for one 3D volume
optimizer = "adam"  # optimizer we want to use, 'adam' or 'momentum'
Nx, Ny = 128, 128 # final size of the image

learning_rate = 0.01
display_step = 50
epochs = 30
training_iters = 200
save_epoch = 1


# here indicating the GPU you want to use. if you don't have GPU, just leave it.
gpu_ind = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ind; # 0,1,2,3


####################################################
####                DATA LOADING                 ###
####################################################

# Specify the location with for the training data #
if platform == "linux" or platform == "linux2":
	train_data_path = './data/data_downconverted'; upscaling=4 # linux
elif platform == "darwin":
	train_data_path = './test'; upscaling=4 # OS X
elif platform == 'win32':
   train_data_path = '.\\data\\data_downconverted_2'; upscaling=2
   train_data_path = '.\\data\\data_downconverted_2'; upscaling=2
   train_data_path = '.\\data\\data_raw'; upscaling=1
   train_data_path = '.\\data\\data_synthetic_2'; upscaling=2
nn_name = 'upsamping_'+str(upscaling)+'_lstm4_'+str(Nx)+'x'+str(Ny)+'_time_'+str(ntimesteps)+'_batchnorm_new2'
                           
# Specify the location with for the validation data #
data_provider = image_util.ImageDataProvider_hdf5_vol(train_data_path, upscaling=upscaling, nchannels = 1, mysize=(Nx, Ny), ntimesteps=ntimesteps)

# Specify the location with for the validation data #
valid_data_path = train_data_path#'./data_32bit_for_unet'
valid_provider = image_util.ImageDataProvider_hdf5_vol(valid_data_path, upscaling=upscaling, nchannels = 1, mysize=(Nx, Ny), ntimesteps=ntimesteps)


####################################################
####                  NETWORK                    ###
####################################################

"""
	here we specify the neural network.

"""

#-- Network Setup --#
# set up args for the unet

net = sofi.SOFI(batchsize=batch_size, Nx=Nx, Ny=Ny, img_channels=1, features_root=features_root, ntimesteps=ntimesteps, cost="mean_squared_error")
       

####################################################
####                 TRAINING                    ###
####################################################
#tensorboard --logdir=.\

# output paths for results
output_root = 'networks/'
output_path = output_root+ nn_name+'_gpu' + gpu_ind + '/models'
prediction_path = output_root + nn_name+'_gpu' + gpu_ind + '/validation'

# make a trainer for muscat
trainer = train.trainer_sofi(net, batch_size=batch_size, optimizer = "adam")

# train 
path = trainer.train(data_provider, output_path, valid_provider, batch_size, training_iters=training_iters, epochs=epochs, display_step=display_step, save_epoch=save_epoch, prediction_path=prediction_path)