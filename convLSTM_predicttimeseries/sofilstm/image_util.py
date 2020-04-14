
'''
author: bdiederich
'''
from __future__ import print_function, division, absolute_import, unicode_literals

#import cv2
import glob
import numpy as np
import h5py
from PIL import Image
import tensorflow as tf
import os
import scipy.io as sio
import NanoImagingPack as nip

class BaseDataProvider(object):
    """
    Abstract base class for DataProvider implementation. Subclasses have to
    overwrite the `_next_file` method that load the next data and label array.
    This implementation automatically clips the data with the given min/max and
    normalizes the values to (0,1]. To change this behavoir the `_process_data`
    method can be overwritten. To enable some post processing such as data
    augmentation the `_post_process` method can be overwritten.
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    """

    channels = 1
    n_class = 1


    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf
        self.x=0
        self.y=0
        self.nb=0

    def _load_data_and_label(self):
        data,label= self._next_file()

        return data, label

    def _process_labels(self, label):
        if self.n_class == 2:
            nx = label.shape[1]
            ny = label.shape[0]
            labels = np.zeros((ny, nx, self.n_class), dtype=np.float32)
            labels[..., 1] = label
            labels[..., 0] = ~label
            return labels

        return label


    def __call__(self, nbatch):
        data,labels = self._next_timestep()

        # allocate memory
        # we concatenated the ntimesteps with nbatches due to the inability of 
        # Tensorflows conv2D layers to read 5D arrays - won't be an issue..
        X = np.expand_dims(data,0)
        Y = np.expand_dims(labels,0)
        
        for i in range(1, nbatch):
            data, labels = self._next_timestep()
                        
            # concatenate batches and timesteps
            data = np.expand_dims(data,0)
            labels = np.expand_dims(labels,0)
            
            X = np.concatenate((X, data))
            Y = np.concatenate((Y, labels))
    
        # Expand-Dims, because we don't have any channels!
        if(len(X.shape)==3):
            X = np.expand_dims(X,-1)
        if(len(Y.shape)==3):
            Y = np.expand_dims(Y,-1)

        return X, Y



class ImageDataProvider_hdf5_vol(BaseDataProvider):
    """
    Generic data provider for images in hdf5 format, supports gray scale and colored images.
    Assumes that the data images and label images are stored in the same folder
    and that the labels have a different file suffix

    Usage:
    data_provider = ImageDataProvider_hdf5("..fishes/train/*.h5",batchsize)

    :param search_path: a glob search pattern to find all data and label images
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping

    """


    def __init__(self, search_path, nchannels=1, mysize=None, ntimesteps=9, upscaling=2, test=False, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf
        self.mysize = mysize
        self.upscaling = int(upscaling)
        
        self.file_idx = -1
        self.timestep_idx = 0
        self.Nz = -1 # number of timesteps/Z-slices in the dataset
        
        self.test=test
        self.channels=nchannels # eventually it can be used for complex data
        self.n_class=1
        self.data_files, self.nfiles = self._get_list(mypath=search_path)
        self.ntimesteps = ntimesteps # corresponds to Nz!
        
        # Declare the names where the GT and labels are stored
        self.gtname = 'sr'
        self.holoname = 'sofi'
        
        assert len(self.data_files) > 0, "No training files"
        print("Number of files used: %s" % len(self.data_files))

        if self.test:
            self.ids=range(0,len(self.data_files))
        else:
            self.ids=np.random.permutation(len(self.data_files))

    
    # returns training data, training labels, testing data, testing labels
    def _get_list(self, mypath='./data'):
        # find all subdirs 

        subfolders = [f.path for f in os.scandir(mypath) if f.is_dir() ]
        
        # Find all sr and sofi files in subdir    
        mysubdirs = []
        for i in range(len(subfolders)):
            for file in os.listdir(subfolders[i]):
                if file.endswith("sofi.mat"):
                    mysubdirs.append(os.path.join(subfolders[i]))#, holoname, '.mat')
                    
        nfiles = int(len(mysubdirs))
        return mysubdirs, nfiles

    def _load_file(self, path, opt):
        h5f = h5py.File(path, 'r')
        x = np.array(h5f[opt])
        return x
    
    def _load_file_mat(self, path, opt):
        mat_contents = sio.loadmat(path)
        x = np.array(mat_contents[opt]) #arrangement is (x,y,z) in order
        return x

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.data_files):
            self.file_idx = -1
            self.ids=np.random.permutation(len(self.data_files))

    def _next_file(self):
        self._cylce_file()
        self.image_name = self.data_files[self.ids[self.file_idx]]
        # print('Now providing next Mat-file: '+self.image_name)
                
        # this is for MAT-files with --v7.3 option
        data = self._load_file_mat(os.path.join(self.image_name, self.holoname+'.mat'), self.holoname)
        label = self._load_file_mat(os.path.join(self.image_name, self.gtname+'.mat'), self.gtname)
        if(not(self.mysize is None)):
				# randomly extract a smaller roi
            if(False):#(data.shape[0]-self.mysize[0])//2)>0:
                mycenter_x = self.mysize[0]//2+np.random.randint(0,(data.shape[0]-self.mysize[0])//2)
                mycenter_y = self.mysize[1]//2+np.random.randint(0,(data.shape[1]-self.mysize[1])//2)
                data = nip.extract(nip.image(data), (self.mysize[0]//self.upscaling,self.mysize[1]//self.upscaling, data.shape[-1]), (mycenter_x//self.upscaling,mycenter_y//self.upscaling))
                label = nip.extract(nip.image(label), (self.mysize[0], self.mysize[1]), (mycenter_x,mycenter_y))
            else:
                data = nip.extract(nip.image(data), (self.mysize[0]//self.upscaling,self.mysize[1]//self.upscaling, data.shape[-1]))
                label = nip.extract(nip.image(label), (self.mysize[0], self.mysize[1]))

        data = self._process_data(data)
        label = self._process_truths(label)

        return data,label

    def _next_timestep(self):
        self.data, self.labels = self._next_file() # load new data 
        
        # # if data is empty or time-steps reached their max, we have to get a new set of data
        # if(self.Nz==-1 or ((self.timestep_idx+self.ntimesteps)>self.Nz)):
        #     self.data, self.labels = self._next_file() # load new data 
        #     self.Nz = self.data.shape[-1] # this is the dimension of the Z-axis/number of timesteps
        #     self.timestep_idx = 0 # reset timestep

        self.timestep_idx = np.random.randint(0,self.data.shape[-1]-self.ntimesteps-1)
        # timestep finds in last dimension, so extract here -> [Nx, Ny, Ntime]
        X_time = self.data[:,:,self.timestep_idx:self.timestep_idx+self.ntimesteps]
        Y_time = self.labels
        
        #self.timestep_idx += 3 # increase the index by two to have more variation in the data
        return X_time,Y_time
    
    def _process_truths(self, truth):
        #print('Preprocessing truth (normalizing)')
        #truth = np.clip(np.fabs(truth), self.a_min, self.a_max)
        #truth -= np.min(truth)
        truth = truth/256. #/np.max(truth)
        #truth -= .5
        return truth

    def _process_data(self, data):
        #print('Preprocessing data (normalizing)')
        #data = np.clip(np.fabs(data), self.a_min, self.a_max)
        #data = nip.resample(data, factors=[2., 2., 1.])
        from skimage.transform import resize
        Nx, Ny, Nz =  data.shape
        data = resize(data, (Nx*self.upscaling, Nx*self.upscaling, Nz))
        #data -= np.min(data)
        #data = data/256. #np.max(data)
        #data -= .5
        return data


