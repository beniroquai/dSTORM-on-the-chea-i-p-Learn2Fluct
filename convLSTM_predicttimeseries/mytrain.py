from muscat.unet_bn import Unet_bn
from muscat.train import Trainer_bn

from muscat import image_util

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

# args for training
batch_size = 1  # batch size for training
ntimesteps = 15  # number of time-steps for one 3D volume
valid_size = batch_size  # batch size for validating (must be same as batch_size!)
optimizer = "adam"  # optimizer we want to use, 'adam' or 'momentum'
Nx, Ny = 128, 128

# here indicating the GPU you want to use. if you don't have GPU, just leave it.
gpu_ind = '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ind; # 0,1,2,3


####################################################
####                DATA LOADING                 ###
####################################################

# Specify the location with for the training data #
if platform == "linux" or platform == "linux2":
	train_data_path = './data_32bit_for_unet' # linux
elif platform == "darwin":
	train_data_path = '/Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/BOSTON/3DNN/DATA/data_32bit/data_32bit_for_unet' # OS X

#data = preprocess(data, data_channels)    # 4 dimension -> 3 dimension if you do data[:,:,:,1]
#truths = preprocess(truths_mat['obGausWeak128'], truth_channels)
data_provider = image_util.ImageDataProvider_hdf5_vol(train_data_path, nchannels = 1, ntimesteps=ntimesteps)
#TODO: Pre and Postprocessing in imageutil?

# Specify the location with for the validation data #
valid_data_path = train_data_path#'./data_32bit_for_unet'
valid_provider = image_util.ImageDataProvider_hdf5_vol(valid_data_path, nchannels = 1, ntimesteps=ntimesteps)


####################################################
####                  NETWORK                    ###
####################################################

"""
	here we specify the neural network.

"""

#-- Network Setup --#
# set up args for the unet
kwargs = {
	"layers": 4,           # how many resolution levels we want to have
	"conv_times": 2,       # how many times we want to convolve in each level
	"features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
	"filter_size": 3,      # filter size used in convolution
	"pool_size": 2,        # pooling size used in max-pooling
	"summaries": True
}


net = Unet_bn(batchsize=batch_size, Nx=Nx, Ny=Ny, ntimesteps=ntimesteps, cost="mean_squared_error", **kwargs)


####################################################
####                 TRAINING                    ###
####################################################


# output paths for results
output_path = 'gpu' + gpu_ind + '/models'
prediction_path = 'gpu' + gpu_ind + '/validation'
# restore_path = 'gpu001/models/50099_cpkt'

# optional args
opt_kwargs = {
		'learning_rate': 0.001
}


# make a trainer for muscat
trainer = Trainer_bn(net, batch_size=batch_size, optimizer = "adam", opt_kwargs=opt_kwargs)

# train 
path = trainer.train(data_provider, output_path, valid_provider, valid_size, training_iters=100, epochs=1000, display_step=20, save_epoch=100, prediction_path=prediction_path)





