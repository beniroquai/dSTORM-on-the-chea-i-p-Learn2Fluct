#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 09:36:23 2018

@author: bene
"""

import numpy as np
import matplotlib.pyplot as plt
import image_util_preprocess
import NanoImagingPack as nip
import sys
import os


Nx, Ny = 128, 128 # final size of the image
Ntime = 25


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
   train_data_path = prefix +'\\data\\data'; upscaling=1

#%
upscaling=2
# Specify the location with for the validation data #
data_provider = image_util_preprocess.ImageDataProvider(train_data_path, nchannels = 1, 
                mysize=(Nx, Ny), Ntime=Ntime, downscaling=upscaling, test=False, \
                n_modes = 50, mode_max_angle = 15, kernel_size = 1, n_photons = 100, n_readnoise = 10, 
                downsampling = 1, kernelsize=1, quality_jpeg=80)

#%%
nimages=4
Ys, X_noisys, X_cleans  = data_provider(nimages)

for i in range(nimages):
    Y, X_noisy, X_clean = Ys[i,], X_noisys[i,], X_cleans[i,]

    # nip.image(np.transpose(np.squeeze(X_noisy),[2,0,1]))
    # nip.image(np.transpose(np.squeeze(X_clean),[2,0,1]))
    # print('X:'+str(X_noisy.shape))
    # print('Y:'+str(Y.shape))
    # #plt.imshow(X[:,64,:])
    
    plt.subplot(221),plt.title('SR'),plt.imshow(np.squeeze(Y)), plt.colorbar()
    plt.subplot(222),plt.title('STD'),plt.imshow(np.squeeze(np.std(X_noisy,-1))),plt.colorbar()
    plt.subplot(223),plt.title('Mean'),plt.imshow(np.squeeze(np.std(X_clean,-1))),plt.colorbar()
    plt.subplot(224),plt.title('RAW'),plt.imshow(np.squeeze(X_clean[:,:,1])),plt.colorbar()
    plt.show()
    
    
import scipy.io as sio
sio.savemat('test.mat', {"sofi": X_noisy})
