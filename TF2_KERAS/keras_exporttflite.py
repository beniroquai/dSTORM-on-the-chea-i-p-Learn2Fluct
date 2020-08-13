# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 08:58:10 2020

@author: diederichbenedict

Some sources:
    https://github.com/mnicholas2019/M202A/blob/master/PreviousWork/Activity-Recognition.ipynb
    https://github.com/vikranth94/Activity-Recognition/blob/master/existing_models.py
"""

import numpy as np

from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.callbacks.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from keras import optimizers

import keras_datagenerator as data
import keras_network as net
import keras_utils as utils
import keras_tensorboard as ktb

import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sys import platform
import io

import matplotlib as mpl
mpl.rc('figure',  figsize=(24, 20))


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


''' start the code here
'''
# Define Parameters
Ntime = 30
Nbatch = 1
Nx = 256    
Ny = 256
upscaling = 2

# Specify the location with for the training data #
if platform == "linux" or platform == "linux2":
    base_dir = './'
    data_dir = base_dir+'data_downconverted'; upscaling=2;
elif platform == "darwin":
	data_dir = './data' # OS X
elif platform == 'win32':
    base_dir = ''#.\\'
    #data_dir = base_dir+'data\\\\data_downconverted_4'; upscaling=4;
    #data_dir = base_dir+'data\\\\data_raw'; upscaling=1;
    data_dir = base_dir+'C:\\Users\\diederichbenedict\\Dropbox\\Dokumente\\Promotion\\PROJECTS\\STORMoChip\\PYTHON\\Learn2Fluct\\convLSTM_predicttimeseries\data\\data'; upscaling=2;


# define the data generator
validation_generator = data.DataGenerator(data_dir, n_batch=Nbatch,
                mysize=(Nx, Ny), n_time=Ntime, downscaling=upscaling, \
                n_modes = 50, mode_max_angle = 15, n_photons = 100, n_readnoise = 10, 
                kernelsize=1, quality_jpeg=80)
        

# test dataloader 
i_testimage = 1
myX,myY = validation_generator.__getitem2D__(i_testimage)
# display images
plt.subplot(131)
plt.title('SR'), plt.imshow(np.squeeze(myY[0,])), plt.colorbar()
plt.subplot(132)
plt.title('mean'), plt.imshow(np.squeeze(np.mean(myX[0,],axis=(0,-1)))), plt.colorbar()
plt.subplot(133)
plt.title('raw'), plt.imshow(np.squeeze(myX[0,0,:,:,0])), plt.colorbar()
plt.show()

## THIS DOES NOT WORK WITH THE TF-GPU VERSION ON WINDOWS!!
model = tf.keras.models.load_model('test.hdf5')
model.summary()

# Predict 
batch_size_test=np.random.randint(0,10)
myX,myY = validation_generator.__getitem__(batch_size_test)
myY_result = model.predict(myX, batch_size=Nbatch)

# Display
myX_reshape = np.reshape(myX, (Nbatch, Ntime, Nx//2, Ny//2, 1))
myY_reshape = np.reshape(myY, (Nbatch, Nx, Ny, 1))
myY_result_reshape = np.reshape(myY_result, (Nbatch, Nx, Ny, 1))

plt.figure()
plt.subplot(221)
plt.title('SOFI'), plt.imshow(np.squeeze(myY_result_reshape[0,]))
plt.subplot(222)
plt.title('SR'), plt.imshow(np.squeeze(myY_reshape[0,]))
plt.subplot(223)
plt.title('Std'), plt.imshow(np.squeeze(np.std(myX_reshape[0,],axis=(0,-1))))
plt.subplot(224)
plt.title('RAW'), plt.imshow(np.squeeze(myX_reshape[0,-1]))
plt.show()
#
#%% export tflite
model = tf.keras.models.load_model('test.hdf5')
model.summary()
converter = tf.lite.TFLiteConverter.from_keras_model(model)
#converter.experimental_new_converter=False
converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()
open('converted_model'+str(Nx)+'_'+str(Ntime)+'_keras.tflite', 'wb').write(tflite_model)



