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
# Specify the location with for the training data #
train_data_path = '..\\data_downconverted'
mysize = (100,100)
#data = preprocess(data, data_channels)    # 4 dimension -> 3 dimension if you do data[:,:,:,1]
#truths = preprocess(truths_mat['obGausWeak128'], truth_channels)
data_provider = image_util.ImageDataProvider_hdf5_vol(train_data_path, mysize = mysize, nchannels = 1, ntimesteps=50)
#data_provider = image_util.ImageDataProvider_hdf5_vol(train_data_path, nchannels = 1, ntimesteps=50)


X, Y = data_provider(1)
nip.image(np.transpose(np.squeeze(X),[2,0,1]))
print('X:'+str(X.shape))
print('Y:'+str(Y.shape))
#plt.imshow(X[:,64,:])

plt.subplot(121),plt.imshow(np.squeeze(Y))
plt.subplot(122),plt.imshow(np.squeeze(np.mean(X,-1)))