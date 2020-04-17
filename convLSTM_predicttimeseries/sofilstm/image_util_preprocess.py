
# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.

'''
author: jakeret
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
import scipy.io as sio
from skimage.draw import line_aa
from scipy.ndimage import gaussian_filter
import tifffile as tif
import matplotlib.pyplot as plt
import cv2
from skimage import transform


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
        gt, result_noisy, result_clean  = self._next_timestep()

        # allocate memory
        # we concatenated the ntimesteps with nbatches due to the inability of 
        # Tensorflows conv2D layers to read 5D arrays - won't be an issue..
        Y = np.expand_dims(gt,0)
        X_noisy = np.expand_dims(result_noisy,0)
        X_clean = np.expand_dims(result_clean,0)
        
        for i in range(1, nbatch):
            gt, result_noisy, result_clean  = self._next_timestep()
                        
            # concatenate batches and timesteps
            gt = np.expand_dims(gt,0)
            result_noisy = np.expand_dims(result_noisy,0)
            result_clean = np.expand_dims(result_clean,0)

            gt = np.expand_dims(np.squeeze(gt),0)
            result_noisy = np.expand_dims(np.squeeze(result_noisy),0)
            result_clean = np.expand_dims(np.squeeze(result_clean),0)
                        
            Y = np.concatenate((Y, gt))
            X_noisy = np.concatenate((X_noisy, result_noisy))
            X_clean = np.concatenate((X_clean, result_clean))
            
    
        # Expand-Dims, because we don't have any channels!
        if(len(Y.shape)==3):
            Y = np.expand_dims(Y,-1)

        return Y, X_noisy, X_clean



class ImageDataProvider(BaseDataProvider):
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


    def __init__(self, search_path, nchannels=1, mysize=None, ntimesteps=9, downscaling=2, test=False, \
                 n_modes = 50, mode_max_angle = 15, kernel_size = 1, n_photons = 100, n_readnoise = 10, \
                 downsampling = 1, kernelsize=1, quality_jpeg=80):
        self.mysize = mysize
        self.downscaling = downscaling
        
        self.file_idx = -1
        self.timestep_idx = 0

        # illumination settings
        self.n_modes = n_modes
        self.mode_max_angle = mode_max_angle
        self.kernelsize = kernelsize
        self.n_photons = n_photons
        self.n_readnoise = n_readnoise
        self.quality_jpeg = 80
        
        
        self.test=test
        self.channels=nchannels # eventually it can be used for complex data
        self.n_class=1
        self.data_files, self.nfiles = self._get_list(mypath=search_path)
        self.ntimesteps = ntimesteps # corresponds to Nz!
        
        # Declare the names where the GT and labels are stored
        self.gtname = 'sr'
        self.holoname = 'sofi'
        
        self.which_channel = 'R'
        
        assert len(self.data_files) > 0, "No training files"
        print("Number of files used: %s" % len(self.data_files))

        if self.test:
            self.ids=range(0,len(self.data_files))
        else:
            self.ids=np.random.permutation(len(self.data_files))
            
        # precompute the illumination pattern
        self.myallilluframes = self._generate_fluctuation_mat()
    
    def _generate_fluctuation_mat(self):
        print("Precompute the illumination pattern")
        myallmodes = np.zeros((self.mysize[0], self.mysize[1], self.ntimesteps*5))
        # generate illuminating modes
        
        for i_frame in range(self.ntimesteps*5):
            for i_modes in range(self.n_modes):
                while(True):
                    start_x = np.random.randint(0,self.mysize[0])
                    start_y = np.random.randint(0,self.mysize[1])
                    # make sure the angle is not too steep!
                    if(np.abs(np.arctan((start_x-start_y)/self.mysize[0])/np.pi*180)<self.mode_max_angle):
                        break
                # https://stackoverflow.com/questions/31638651/how-can-i-draw-lines-into-numpy-arrays            
                rows, cols, weights = line_aa(start_x, 0, start_y, self.mysize[0]-1)    # antialias line
                myallmodes[rows, cols, i_frame] = np.random.randint(20,90)/100
        return myallmodes #plt.imshow(np.mean(myallmodes,-1)), plt.show()
        # print('Frame: '+str(i_frame))

    def _simulate_microscope(self, data):
        # split image and select the channel of choisce
        try:
            if self.which_channel=='R':
                    mysample = data[:,:,0]
            elif self.which_channel=='G':
                    mysample = data[:,:,1]
            elif self.which_channel=='B':
                    mysample = data[:,:,2]            
        except:
            mysample = data
                
        # ---------------------------------------------------
        # produce some SOFI data for the on-chip simulation
        # ---------------------------------------------------
    
        # normalize the sample
        # mysample = nip.resample(mysample, factors =(.5,.5))
        mysample = nip.extract(mysample, self.mysize)
        mysample -= np.min(mysample)
        mysample = mysample/np.max(mysample)*self.n_photons


        # allocate some memory
        myresultframe_noisy = np.zeros((self.mysize[0]//self.downscaling, self.mysize[1]//self.downscaling, self.ntimesteps))
        myresultframe_clean = np.zeros((self.mysize[0]//self.downscaling, self.mysize[1]//self.downscaling, self.ntimesteps))
        # iterate over all frames
        
        
        for iframe in range(self.ntimesteps):
            # generate illumination pattern by randomly selecting illumination frames
            myillutmp = self.myallilluframes[:,:,np.random.randint(0, self.ntimesteps)]
                        
            # illuminate the sample with the structured illumination 
            myresultframe = myillutmp*mysample
            # subsample data
            myresultframe = nip.resample(myresultframe, [1/self.downscaling, 1/self.downscaling])/self.downscaling
            myresultframe -= np.min(myresultframe)# handle zeros
            
            myresultframe_clean[:,:,iframe] = (myresultframe)
            myresultframe = gaussian_filter(myresultframe, sigma=self.kernelsize)
            
            # add noise
            myresultframe = nip.noise.poisson(nip.image(myresultframe), self.n_photons)
            myresultframe = myresultframe + self.n_readnoise*np.random.randn(myresultframe.shape[0],myresultframe.shape[1])
            

            # add compression artifacts 
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality_jpeg]
            result, encimg = cv2.imencode('.jpg', myresultframe, encode_param)
            myresultframe_compressed = np.mean(cv2.imdecode(encimg, 1),-1)
            myresultframe_noisy[:,:,iframe] = myresultframe_compressed#nip.resample(myresultframe_compressed, [self.downscaling, self.downscaling])/self.downscaling

            
        return mysample, myresultframe_noisy, myresultframe_clean
    
    # returns training data, training labels, testing data, testing labels
    def _get_list(self, mypath='./data'):
        # find all subdirs 

        # Find all sr and sofi files in subdir    
        mysamplefiles = []
        for file in glob.glob(os.path.join(mypath, "*.tif")):
            mysamplefiles.append(os.path.join(file))
                    
        nfiles = int(len(mysamplefiles))
        return mysamplefiles, nfiles

    def _load_file(self, path):
        if(np.random.randint(0,10)<3):
            # generate simulated data
            if(np.random.randint(0,2)):
                SizePar=np.random.randint(1,3)
                Ngraphs = np.random.randint(3,14)
                Maxtimesteps=50
                mytif = self._simulateactin(Ngraphs=Ngraphs, SizePar=SizePar, Maxtimesteps=Maxtimesteps)
            else:
                NEcolis = np.random.randint(60,140)
                mytif = self._simulateecoli(NEcolis=NEcolis)
        else:
            mytif = tif.imread(path)
        mytif -= np.min(mytif)
        mytif /= np.max(mytif)

        if(np.random.randint(0,2)):
            # random flips 
            #print('Im applying flipping')
            mytif = np.flip(mytif, axis=np.random.randint(0,2))
        if(np.random.randint(0,2)):
            # random flips 
            #print('Im applying flipping')
            mytif = np.fliplr(mytif)    
            
        if (False):#np.random.randint(0,2)):
            # random affine transformations 
            
            myrot = np.random.randint(0,360)/360*np.pi
            myscale = np.random.randint(100,150)/100
            mytranslation = (np.random.randint(0,40),np.random.randint(0,40))
            
            #print('Im applying affine transformation:'+str(myrot)+' - '+str(myscale)+' - '+str(mytranslation))
            myaffine = transform.AffineTransform(scale=(myscale,myscale), rotation=myrot,translation=mytranslation)
            mytif = transform.warp(mytif, myaffine.inverse)

        return mytif
    
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
        label = self._load_file(os.path.join(self.image_name))
     
        # simulate microscpe 
        mysample, myresultframe_noisy, myresultframe_clean = self._simulate_microscope(label)
        
        mysample = self._preprocess(mysample)
        myresultframe_noisy = self._preprocess(myresultframe_noisy)
        myresultframe_clean = self._preprocess(myresultframe_clean)
   
        return mysample,myresultframe_noisy, myresultframe_clean

    def _next_timestep(self):
        self.gt, self.result_noisy, self.result_clean = self._next_file() # load new data 
               
        #self.timestep_idx += 3 # increase the index by two to have more variation in the data
        return self.gt, self.result_noisy, self.result_clean 
    
    def _preprocess(self, truth):
        #print('Preprocessing truth (normalizing)')
        #truth = np.clip(np.fabs(truth), self.a_min, self.a_max)
        truth -= np.min(truth)
        truth = truth/np.max(truth)
        #truth -= .5
        return truth
    
  
    def _simulateactin(self, Ngraphs=10, SizePar = 2, Maxtimesteps=50):
        # https://ipython-books.github.io/133-simulating-a-brownian-motion/
        # We add 10 intermediary points between two
        # successive points. We interpolate x and y.
        
        Nx, Ny = self.mysize[0]*2, self.mysize[1]*2      
        
        myresult = np.zeros((Nx, Ny))
        
        for igraphs in range(Ngraphs):
            mytimesteps=np.random.randint(20,Maxtimesteps)
            x = np.cumsum(np.random.randn(mytimesteps))
            y = np.cumsum(np.random.randn(mytimesteps))
            
            x2 = np.interp(np.arange(mytimesteps*SizePar), np.arange(mytimesteps)*SizePar, x)
            y2 = np.interp(np.arange(mytimesteps*SizePar), np.arange(mytimesteps)*SizePar, y)
            
            x2-=np.min(x2)
            y2-=np.min(y2)
            x2/= np.max(x2)
            y2/= np.max(y2)
            x2 = np.int32(x2*(Nx-1))
            y2 = np.int32(y2*(Ny-1))
            
        
            
            for ii in range(1,mytimesteps):
                rows, cols, weights = line_aa(x2[ii], y2[ii], x2[ii-1], y2[ii-1])    # antialias line
                myresult[rows, cols] = np.random.randint(30,90)/100
        myresult = nip.resample(myresult, (.5,.5))      
        myresult /= np.max(myresult)  
        
        return myresult


    def _simulateecoli(self, NEcolis=10):
         
        Nx, Ny = self.mysize[0]*2, self.mysize[1]*2      
        myresult = np.zeros((Nx*2, Ny*2))
        

        myresult = np.zeros((Nx, Ny))
        mymaxlength = 20
        
        for igraphs in range(NEcolis):        
            myx1 = np.random.randint(0,Nx)
            myy1 = np.random.randint(0,Ny)
            myx2 = myx1 +np.random.randint(-mymaxlength/2,mymaxlength/2)
            myy2 = myy1 +np.random.randint(-mymaxlength/2,mymaxlength/2)
        
            rows, cols, weights = line_aa(myx1, myy1, myx2, myy2)    # antialias line
            try:
                myresult[rows, cols] = np.random.randint(30,90)/100
            except:
                None
        myresult = nip.resample(myresult, (.5,.5))      
        myresult /= np.max(myresult)  
        return myresult