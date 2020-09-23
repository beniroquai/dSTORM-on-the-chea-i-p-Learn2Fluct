# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 08:58:10 2020

@author: diederichbenedict

Some sources:
    https://github.com/mnicholas2019/M202A/blob/master/PreviousWork/Activity-Recognition.ipynb
    https://github.com/vikranth94/Activity-Recognition/blob/master/existing_models.py

"""

import numpy as np
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from tensorflow.keras import optimizers

import keras_datagenerator as data
import keras_network as net

import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt

from datetime import datetime
from sys import platform
import io
import tifffile as tif
import matplotlib as mpl
mpl.rc('figure',  figsize=(24, 20))


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


''' Some functions for displaying training progress
'''


''' start the code here
'''
#%% Define Parameters
Ntime = 25
Nbatch = 1
Nx = 256    
Ny = 256
features = 1
Nchannel = 1
model_dir = "logs_bilinear/"# + datetime.now().strftime("%Y%m%d-%H%M%S")
network_type = 'SOFI_ConvLSTM2D'
data_dir = 'C:\\Users\\diederichbenedict\\Dropbox\\Dokumente\\Promotion\\PROJECTS\\STORMoChip\\PYTHON\\Learn2Fluct\\convLSTM_predicttimeseries\data\\data'; 
upscaling=2;
export_type = 'tfjs' #'tflite' # or 'tfjs'


# define the data generator
data_generator = data.DataGenerator(data_dir, n_batch=Nbatch,
                mysize=(Nx, Ny), n_time=Ntime, downscaling=upscaling, \
                n_modes = 50, mode_max_angle = 15, n_photons = 100, n_readnoise = 10, 
                kernelsize=1, quality_jpeg=80,
                export_type=export_type)

    

#%%
# Display
batch_size_test = 551
myX,myY = data_generator.__getitem__(batch_size_test)


plt.figure()
plt.subplot(131)
plt.title('SR'), plt.imshow(np.squeeze(myY[0,]))#, plt.colorbar()
plt.subplot(132)
plt.title('Single Frame'), plt.imshow(np.squeeze(myX)[0])#, plt.colorbar()
plt.subplot(133)
plt.title('Mean'), plt.imshow(np.squeeze(np.mean(myX[0,],axis=(0,-1))))#, plt.colorbar()
plt.show()

#%%
# Save
tif.imsave('./testdata/01_artificial_sofi_timeseries_txy.tif',np.squeeze(myX))
tif.imsave('./testdata/01_artificial_sofi_groundtruth_txy.tif',np.squeeze(myY))
