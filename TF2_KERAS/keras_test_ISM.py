# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 08:58:10 2020

@author: diederichbenedict

Some sources:
    https:\\github.com/mnicholas2019/M202A/blob/master/PreviousWork/Activity-Recognition.ipynb
    https:\\github.com/vikranth94/Activity-Recognition/blob/master/existing_models.py

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

import NanoImagingPack as nip

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


''' start the code here
'''
# Define Parameters
n_pix_on = 1
n_pix_off = 9
n_shift = 1
n_period = (n_pix_on+n_pix_off)//n_shift
n_gauss_illu = 2
Ntime = n_period**2
n_gauss_det = 1

n_photon =  150
Nbatch = 1
Nfilterlstm = 16
Nx = 256    
Ny = 256
features = 1
Nchannel = 1
model_dir = "logs/"# + datetime.now().strftime("%Y%m%d-%H%M%S")
network_type = 'SOFI_ConvLSTM2D'
export_type = 'tfjs' #'tflite' # or 'tfjs'

# Training parameters
Nepochs = 150
Niter = 100


# Specify the location with for the training data #
if platform == "linux" or platform == "linux2":
    data_dir = '../data/data';
    upscaling=2 # linux
elif platform == "darwin":
	train_data_path = './test' # OS X
elif platform == 'win32':
    base_dir = ''#.\\'
    #data_dir = base_dir+'data\\\\data_downconverted_4'; upscaling=4;
    #data_dir = base_dir+'data\\\\data_raw'; upscaling=1;
    data_dir = base_dir+'C:\\Users\\diederichbenedict\\Dropbox\\Dokumente\\Promotion\\PROJECTS\\STORMoChip\\PYTHON\\Learn2Fluct\\convLSTM_predicttimeseries\data\\data'; upscaling=2;


from tensorflow import keras

# define the data generator
training_generator = data.DataGenerator(data_dir, n_batch=Nbatch,
                mysize=(Nx, Ny), n_time=Ntime, downscaling=upscaling, \
                n_photons = n_photon, n_readnoise = 5, 
                kernelsize=n_gauss_det, quality_jpeg=100,
                export_type=export_type,
                illumination_type = "ISM",
                n_pix_on = n_pix_on,
                n_pix_off = n_pix_off,
                n_gauss_illu = n_gauss_illu, 
                n_shift = n_shift)
#%% test dataloader 
i_testimage=np.random.randint(0,training_generator.nfiles)
myX,myY = training_generator.__getitem2D__(i_testimage)
# display images
plt.subplot(131)
plt.title('SR'), plt.imshow(np.squeeze(myY[0,]),cmap='gray'), plt.colorbar()
plt.subplot(132)
plt.title('mean'), plt.imshow(np.squeeze(np.mean(myX[0,],axis=(0,-1))),cmap='gray'), plt.colorbar()
plt.subplot(133)
plt.title('raw'), plt.imshow(np.squeeze(myX[0,0,:,:,0]),cmap='gray'), plt.colorbar()
plt.show()
tif.imsave('teststack.tif', np.squeeze(myX))
plt.imshow(np.log(1+np.abs(nip.ft(np.squeeze(myX)[1,:,:]))))


#%%
model = keras.models.load_model('./ISM/test.hdf5')
model.summary()

myfilename = './ISM/2020_10_20_ISM_60x_NA1.tif'
myfilename = './ISM/23.10.2020_800x600_HeLa_SiR_newSample_3.tif'

myX_raw = np.float32(tif.imread(myfilename))
myX_real = nip.resample(myX_raw,factors=(1,.22,.22))
myX_real = nip.extract(myX_real, (Ntime,Nx//2,Ny//2))
myX_real = np.expand_dims(np.expand_dims(myX_real,0),-1)


myY_result = model.predict(myX_real, batch_size=Nbatch)

# in case we have the tflite model
if(myY_result.shape[0]==1): myY_result=np.reshape(myY_result,(Nx,Ny))

plt.figure()
plt.subplot(231)
plt.title('ISM'), plt.imshow(np.squeeze(myY_result), cmap='gray'), plt.colorbar()
plt.subplot(232)
plt.title('Mean'), plt.imshow(np.squeeze(np.mean(myX_real[0,],axis=(0,-1))), cmap='gray'), plt.colorbar()#, plt.colorbar()
plt.subplot(233)
plt.title('STDV'), plt.imshow(np.squeeze(np.std(myX_real[0,],axis=(0,-1))), cmap='gray'), plt.colorbar()#, plt.colorbar()
plt.subplot(234)
plt.title('RAW ft'), plt.imshow(np.squeeze(myX_real[:,4,:,:,0]), cmap='gray'), plt.colorbar()#, plt.colorbar()
plt.imshow(np.log(1+np.abs(nip.ft(np.squeeze(myX_real)[1,:,:]))))
plt.subplot(235)
plt.title('RAW'), plt.imshow(np.squeeze(myX_real[:,4,:,:,0]), cmap='gray'), plt.colorbar()#, plt.colorbar()
plt.imshow(np.squeeze(myX_real)[1,:,:])
plt.subplot(236)
plt.title('RAW artificial '), plt.imshow(np.squeeze(myX_real[:,4,:,:,0]), cmap='gray'), plt.colorbar()#, plt.colorbar()
plt.imshow(np.squeeze(myX[0,0,:,:,0]))


plt.subplot(121)
plt.title('RAW ft'), plt.imshow(np.squeeze(myX_real[:,4,:,:,0]), cmap='gray')#, plt.colorbar()
plt.imshow(np.log(1+np.abs(nip.ft(np.squeeze(myX_real)[1,:,:]))))
plt.subplot(122)
plt.imshow(np.log(1+np.abs(nip.ft(np.squeeze(myX)[1,:,:]))))
plt.show()
