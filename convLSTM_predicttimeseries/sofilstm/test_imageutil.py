#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 09:36:23 2018

@author: bene
"""

import numpy as np
import matplotlib.pyplot as plt
import image_util
# Specify the location with for the training data #
train_data_path = '/Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/BOSTON/3DNN/DATA/data_32bit/data_32bit_for_unet'
#data = preprocess(data, data_channels)    # 4 dimension -> 3 dimension if you do data[:,:,:,1]
#truths = preprocess(truths_mat['obGausWeak128'], truth_channels)
data_provider = image_util.ImageDataProvider_hdf5_vol(train_data_path, nchannels = 1, ntimesteps=9)

for i in range(50):
    X, Y = data_provider(1)
    plt.imshow(np.squeeze(np.mean(X, axis=1))), plt.show()


#plt.imshow(X[:,64,:])