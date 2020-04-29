#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 09:36:23 2018

@author: bene
"""

import numpy as np
import matplotlib.pyplot as plt
import image_util
import NanoImagingPack as nip
import sys
import os


Nx, Ny = 128, 128 # final size of the image
ntimesteps = 10


####################################################
####                DATA LOADING                 ###
####################################################

# Specify the location with for the training data #
if sys.platform == "linux" or sys.platform == "linux2":
	train_data_path = './data/data_downconverted'; upscaling=4 # linux
elif sys. platform == "darwin":
	train_data_path = './test'; upscaling=4 # OS X
elif sys.platform == 'win32':
   train_data_path = '.\\data\\data_downconverted_2'; upscaling=2
   train_data_path = '.\\data\\data_raw'; upscaling=1
   prefix = 'C:\\Users\\diederichbenedict\\Dropbox\\Dokumente\\Promotion\\PROJECTS\\STORMoChip\\PYTHON\\Learn2Fluct\\convLSTM_predicttimeseries'
   train_data_path = prefix +'\\data\\data_raw'; upscaling=1
                    
# Specify the location with for the validation data #
data_provider = image_util.ImageDataProvider_hdf5_vol(train_data_path, upscaling=upscaling, nchannels = 1, mysize=(Nx, Ny), ntimesteps=ntimesteps)


#%%

X, Y = data_provider(1)
nip.image(np.transpose(np.squeeze(X),[2,0,1]))
print('X:'+str(X.shape))
print('Y:'+str(Y.shape))
#plt.imshow(X[:,64,:])

plt.subplot(121),plt.imshow(np.squeeze(Y))
plt.subplot(122),plt.imshow(np.squeeze(np.std(X,-1)))