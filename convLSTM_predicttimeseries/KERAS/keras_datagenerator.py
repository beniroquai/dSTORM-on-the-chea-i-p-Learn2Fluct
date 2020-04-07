# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 08:56:53 2020

@author: diederichbenedict
"""

import numpy as np
import keras
import os
from scipy.io import loadmat
import NanoImagingPack as nip

import matplotlib.pyplot as plt

"""
#### Keras Data Generators
To feed the scalograms images to the model we also need to create a custom
data_generator. Data_generators are used to load input data from the drive
on small batches as needed when training the model. That way we avoid
running out of RAM memory when working with large data sets. The generators
are defined on the "**data_generator_classes.py**" file. 
More info about keras data generators:
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""


class DataGenerator(keras.utils.Sequence):
    """ Generates input data for the Keras models """

    def __init__(self, batch_size, dim, data_dir, time_steps, n_channels=1, shuffle=True, upscaling=1):
        """ Initialization """
        self.Nx, self.Ny = dim[0],dim[1]
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.time_steps = time_steps
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.upscaling = upscaling
        self.label_name = 'sr'
        self.input_name = 'sofi'
        
        # find all subfolders
        self.list_folders = [ f.path for f in os.scandir(self.data_dir) if f.is_dir() ]
        print('We have '+str(len(self.list_folders))+' dataset(s).')

        self.on_epoch_end()
        
        
    def __len__(self):
        """ Denotes the number of batches per epoch """
        return int(np.floor(len(self.list_folders) / self.batch_size))

    def __getitem__(self, index):
        """ Generate one batch of data """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_folders_temp = [self.list_folders[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_folders_temp)

        return X, y

    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        self.indexes = np.arange(len(self.list_folders))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_folders_temp):
        """ Generates data containing batch_size samples """
        # Initialization
        X = np.zeros((self.batch_size, self.time_steps, self.Nx, self.Ny, self.n_channels))
        y = np.zeros((self.batch_size, self.Nx, self.Ny, self.n_channels))

        # Generate data
        for ifiles in range(len(list_folders_temp)):
            # Store sample

            # Load mat-files SOFI data: batch, time, nx,ny, channels
            Xtmp = loadmat(os.path.join(list_folders_temp[ifiles], self.input_name+'.mat'))[self.input_name]
            from skimage.transform import resize
            Nx, Ny, Nz =  Xtmp.shape
            Xtmp = resize(Xtmp, (Nx*self.upscaling, Nx*self.upscaling, Nz))*3
            X[ifiles,:,:,:,:] = np.transpose(np.expand_dims(nip.extract(Xtmp, [self.Nx, self.Ny, self.time_steps]),-1),(2,0,1,3))

            # Load mat-files, GT data : batch, nx, ny, channels
            Ytmp = loadmat(os.path.join(list_folders_temp[ifiles], self.label_name+'.mat'))[self.label_name]/2**8
            y[ifiles,:,:,:] = np.expand_dims(nip.extract(Ytmp,[self.Nx, self.Ny]),-1)
            
        X-=np.min(X)
        y-=np.min(y)
        X/=np.max(X)
        y/=np.max(y)
        
        return X, y
