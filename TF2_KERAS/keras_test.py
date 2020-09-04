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

import matplotlib as mpl
mpl.rc('figure',  figsize=(24, 20))


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


''' Some functions for displaying training progress
'''

def plot_to_image():
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.show()
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def log_images(epoch, logs):
    
    # Use the model to predict the values from the validation dataset.
    myX,myY = validation_generator.__getitem__(np.random.randint(100))
    myY_result = model.predict(myX)

    # Display
    myX_reshape = np.reshape(myX, (Nbatch, Ntime, Nx//2, Ny//2, 1))
    myY_reshape = np.reshape(myY, (Nbatch, Nx, Ny, 1))
    myY_result_reshape = np.reshape(myY_result, (Nbatch, Nx, Ny, 1))

    plt.figure()
    plt.subplot(221)
    plt.title('SOFI'), plt.imshow(np.squeeze(myY_result_reshape[0,]),cmap='gray')
    plt.subplot(222)
    plt.title('SR'), plt.imshow(np.squeeze(myY_reshape[0,]),cmap='gray')
    plt.subplot(223)
    plt.title('Mean'), plt.imshow(np.squeeze(np.mean(myX_reshape[0,],axis=(0,-1))),cmap='gray')
    plt.subplot(224)
    plt.title('STD'), plt.imshow(np.squeeze(np.std(myX_reshape[0,],axis=(0,-1))),cmap='gray')
    cm_image = plot_to_image()
      
    # Log the confusion matrix as an image summary.
    with file_writer_cm.as_default():
        tf.summary.image("Reconstructed Results", cm_image, step=epoch)


''' start the code here
'''
#%% Define Parameters
Ntime = 25
Nbatch = 1
Nfilterlstm = 8
Nx = 256    
Ny = 256
features = 1
Nchannel = 1
model_dir = "logs_bilinear/"# + datetime.now().strftime("%Y%m%d-%H%M%S")
network_type = 'SOFI_ConvLSTM2D'

# Training parameters
Nepochs = 100
Niter = 1000


# Specify the location with for the training data #
if platform == "linux" or platform == "linux2":
    base_dir = './'
    data_dir = base_dir+'data_downconverted'; upscaling=2;
elif platform == "darwin":  # OS X
    data_dir = './data'
    upscaling=2
elif platform == 'win32':
    base_dir = ''#.\\'
    #data_dir = base_dir+'data\\\\data_downconverted_4'; upscaling=4;
    #data_dir = base_dir+'data\\\\data_raw'; upscaling=1;
    data_dir = base_dir+'C:\\Users\\diederichbenedict\\Dropbox\\Dokumente\\Promotion\\PROJECTS\\STORMoChip\\PYTHON\\Learn2Fluct\\convLSTM_predicttimeseries\data\\data'; 
    upscaling=2;


# define the data generator
training_generator = data.DataGenerator(data_dir, n_batch=Nbatch,
                mysize=(Nx, Ny), n_time=Ntime, downscaling=upscaling, \
                n_modes = 50, mode_max_angle = 15, n_photons = 100, n_readnoise = 10, 
                kernelsize=1, quality_jpeg=80)
validation_generator = data.DataGenerator(data_dir, n_batch=Nbatch,
                mysize=(Nx, Ny), n_time=Ntime, downscaling=upscaling, \
                n_modes = 50, mode_max_angle = 15, n_photons = 100, n_readnoise = 10, 
                kernelsize=1, quality_jpeg=80)
        

from tensorflow import keras
model = keras.models.load_model('test.hdf5')

# Predict 
batch_size_test=np.random.randint(0,300)
myX,myY = training_generator.__getitem__(batch_size_test)
myY_result = model.predict(myX, batch_size=Nbatch)

# Display
myX_reshape = np.reshape(myX, (Nbatch, Ntime, Nx//2, Ny//2, 1))
myY_reshape = np.reshape(myY, (Nbatch, Nx, Ny, 1))
myY_result_reshape = np.reshape(myY_result, (Nbatch, Nx, Ny, 1))

plt.figure()
plt.subplot(131)
plt.title('SOFI'), plt.imshow(np.squeeze(myY_result_reshape[0,]))#, plt.colorbar()
plt.subplot(132)
plt.title('SR'), plt.imshow(np.squeeze(myY_reshape[0,]))#, plt.colorbar()
plt.subplot(133)
plt.title('Mean'), plt.imshow(np.squeeze(np.mean(myX_reshape[0,],axis=(0,-1))))#, plt.colorbar()
plt.show()
