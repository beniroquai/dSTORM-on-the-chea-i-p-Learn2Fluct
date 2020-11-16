# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 08:56:53 2020

@author: diederichbenedict
"""

import numpy as np
import tensorflow.keras as keras
import os
from scipy.io import loadmat
import NanoImagingPack as nip
import glob
import cv2
import matplotlib.pyplot as plt
from skimage.draw import line_aa
from scipy.ndimage import gaussian_filter
import tifffile as tif
from skimage import transform


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

    def __init__(self, search_path, mysize=None, n_batch = 1, n_time=9, downscaling=2, test=False, \
                     n_modes = 50, mode_max_angle = 15, kernelsize = 1, n_photons = 100, n_readnoise = 10, \
                     quality_jpeg=80, max_background= 3, shuffle=True, export_type = 'tflite', illumination_type='TIRF',
                     n_pix_on = 1, n_pix_off = 9, n_gauss_illu = 2, n_shift=2):
        """ Initialization """
        self.mysize = mysize
        self.Nx, self.Ny = mysize[0],mysize[1]
        self.downscaling = downscaling
        self.n_batch = n_batch
        self.shuffle = shuffle
        if export_type=='tflite': self.reshape = True
        elif export_type=='tfjs': self.reshape = False
        
        self.file_idx = -1
        self.timestep_idx = 0

        # illumination settings
        self.n_modes = n_modes
        self.mode_max_angle = mode_max_angle
        self.kernelsize = kernelsize
        self.n_photons = n_photons
        self.n_readnoise = n_readnoise
        self.quality_jpeg = 80
        self.max_background = max_background
        self.illumination_type = illumination_type
        
        # ISM 
        self.n_pix_on = n_pix_on            # number of pixels which are on in rectangular grid
        self.n_pix_off = n_pix_off          # number of pixels which are off in rectangular grid
        self.n_gauss_illu = n_gauss_illu    # psf of illuminatoin
        self.n_shift = n_shift              # distance between peaks in pixels 

        self.test=test
        self.n_class=1
        self.data_files, self.nfiles = self._get_list(mypath=search_path)
        self.n_time = n_time # corresponds to Nz!

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
        self.myallilluframes = self._generate_illumination_intensity_map()


        self.on_epoch_end()
        
        
    def __len__(self):
        """ Denotes the number of batches per epoch """
        return int(np.floor(len(self.data_files) / self.n_batch))

    def __getitem__(self, index):
        """ Generate one batch of data """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.n_batch:(index+1)*self.n_batch]

        # Find list of IDs
        data_files_temp = [self.data_files[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(data_files_temp)

        return X, y

    def __getitem2D__(self, index):
        """ Generate one batch of data """
        
        # Generate data
        X, y = self.__data_generation(index)

        return X, y
    
    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        self.indexes = np.arange(len(self.data_files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, data_files_temp):
        """ Generates data containing n_batch samples """
        myally = []
        myallX = []
        
        if(self.n_batch>0):
            for files in range(self.n_batch):
                y,X,_ = self._next_file()
                # adapt to Keras framework:
                X = np.transpose(X, [-1,0,1])
                myally.append(y)
                myallX.append(X)
        else: 
            y,X,_ = self._next_file()
            # adapt to Keras framework:
            X = np.transpose(X, [-1,0,1])
            myally = np.expand_dims(y,0)
            myallX = np.expand_dims(X,0)
            
        
        myallX = np.expand_dims(np.array(myallX),-1)
        myally = np.expand_dims(np.array(myally), -1)

        # reshape for the use with tflite
        if self.reshape:
            myallX = np.reshape(myallX, (self.n_batch, self.Nx*self.Ny*self.n_time//(self.downscaling*2)))
            myally = np.reshape(myally, (self.n_batch, self.Nx*self.Ny))
            
        
        # normalize data 0..1
        myallX = myallX-np.min(myallX)
        myallX = myallX/np.max(myallX)
        
        myally = myally-np.min(myally)
        myally = myally/np.max(myally)
        
        return myallX, myally
    

    def _generate_illumination_intensity_map(self):
        
        if self.illumination_type == 'TIRF':
            print("Precompute the illumination pattern for TIRF ")
            n_border = 20 # avoid zero-space in the middle of the pattern
            myallmodes = np.zeros((self.mysize[0]//self.downscaling+n_border, self.mysize[1]//self.downscaling+n_border, self.n_time*5))
            # generate illuminating modes
            for i_frame in range(self.n_time*5):
                for i_modes in range(self.n_modes):
                    while(True):
                        start_x = np.random.randint(0,self.mysize[0]//self.downscaling+n_border)
                        start_y = np.random.randint(0,self.mysize[1]//self.downscaling+n_border)
                        # make sure the angle is not too steep!
                        if(np.abs(np.arctan((start_x-start_y)/(self.mysize[0]//self.downscaling+n_border))/np.pi*180)<self.mode_max_angle):
                            break
                    # https://stackoverflow.com/questions/31638651/how-can-i-draw-lines-into-numpy-arrays
                    rows, cols, weights = line_aa(start_x, 0, start_y, self.mysize[0]//self.downscaling+n_border-1)    # antialias line
                    myallmodes[rows, cols, i_frame] = np.random.randint(20,90)/100
            return nip.extract(myallmodes, (self.mysize[0], self.mysize[1], self.n_time*5)) #plt.imshow(np.mean(myallmodes,-1)), plt.show()
        elif self.illumination_type == "ISM":
            print("Precompute the illumination pattern for ISM")
            n_unitcell = self.n_pix_off+self.n_pix_on
            n_period = n_unitcell//self.n_shift
            if not self.n_time == n_period**2:
                print("ADjust the timesteps to be : "+str(n_period**2))
                error
            

            ism_pattern_all = []            
            for ix in range(0,n_unitcell,self.n_shift):
                for iy in range(0,n_unitcell,self.n_shift):
                    ism_unitcell = np.zeros((n_unitcell, n_unitcell))
                    ism_unitcell[ix:ix+self.n_pix_on, iy:iy+self.n_pix_on] = 1
                    ism_pattern = np.tile(ism_unitcell, [self.mysize[0]//self.downscaling//n_unitcell+1,self.mysize[1]//self.downscaling//n_unitcell+1])
                    ism_pattern = ism_pattern[:self.mysize[0]//self.downscaling,:self.mysize[1]//self.downscaling]
                    ism_pattern_all.append(nip.gaussf(ism_pattern, self.n_gauss_illu))


            return np.transpose(np.array(ism_pattern_all),(1,2,0))
                    
            

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

        # downsample data         
        mysample_sub = cv2.resize(mysample, dsize=None, fx=1/self.downscaling, fy=1/self.downscaling)
        
        # normalize the sample
        #mysample_sub -= np.min(mysample_sub)
        mysample_sub /= np.max(mysample_sub)
       
        # add random backround
        mybackground = np.random.rand(mysample_sub.shape[0],mysample_sub.shape[1])>.99
        mybackground = nip.gaussf(mybackground*1., 10)
        mybackground -= np.min(mybackground)
        mybackground /= np.max(mybackground)
        mybackground *= np.random.randint(20,30)*0.01
        
        mysample_sub = np.array(mysample_sub) + mybackground
        mysample_sub = mysample_sub/np.max(mysample_sub)*self.n_photons

        # allocate some memory
        myresultframe_noisy = np.zeros((self.mysize[0]//self.downscaling, self.mysize[1]//self.downscaling, self.n_time))
        myresultframe_clean = np.zeros((self.mysize[0]//self.downscaling, self.mysize[1]//self.downscaling, self.n_time))
        # iterate over all frames

        # randomize illumination pattern (?)
        myilluorder = np.arange(0,self.n_time)
        np.random.shuffle(myilluorder)

        for iframe in range(self.n_time):
            # generate illumination pattern by randomly selecting illumination frames
            if self.illumination_type == "TIRF":
                myillutmp = self.myallilluframes[:,:,np.random.randint(0, self.n_time)]
            else:
                # preserve order for ISM / SIM 
                myillutmp = self.myallilluframes[:,:,myilluorder[iframe]]

            # illuminate the sample with the structured illumination    
            myresultframe = myillutmp*mysample_sub
            
            #myresultframe = nip.resample(myresultframe, [1/self.downscaling, 1/self.downscaling])/self.downscaling
            myresultframe -= np.min(myresultframe)# handle zeros


            myresultframe_clean[:,:,iframe] = (myresultframe) + np.random.randint(0,self.max_background) # add background

            #convolve with PSF
            myresultframe = gaussian_filter(myresultframe, sigma=self.kernelsize)

            # add noise
            myresultframe = nip.noise.poisson(nip.image(myresultframe), self.n_photons)
            myresultframe = myresultframe + self.n_readnoise*np.random.randn(myresultframe.shape[0],myresultframe.shape[1])


            # add compression artifacts
            if self.quality_jpeg<100:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality_jpeg]
                result, encimg = cv2.imencode('.jpg', myresultframe, encode_param)
                myresultframe_compressed = np.mean(cv2.imdecode(encimg, 1),-1)
            else:
                myresultframe_compressed = myresultframe
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
                mytif = self._simulateactin(Ngraphs=Ngraphs, SizePar=SizePar, Maxtimesteps=Maxtimesteps)*2**8
            else:
                NEcolis = np.random.randint(60,140)
                mytif = self._simulateecoli(NEcolis=NEcolis)*2**8
        else:
            mytif = tif.imread(path)
            mytif = cv2.resize(mytif, dsize=None, fx=.75, fy=.75)
            myX_tmp = mytif.shape[0]
            myY_tmp = mytif.shape[1]
            diffX_tmp = np.floor(np.abs(self.mysize[0]-myX_tmp)/2)
            diffY_tmp = np.floor(np.abs(self.mysize[1]-myY_tmp)/2)
            shiftX_tmp = np.random.randint(-diffX_tmp, diffX_tmp)
            shiftY_tmp = np.random.randint(-diffY_tmp, diffX_tmp)

            mytif = nip.extract(mytif, self.mysize, (myX_tmp//2+shiftX_tmp,myY_tmp//2+shiftY_tmp))

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
        while(True):
            # make sure, that there is some content in the image
            self._cylce_file()
            self.image_name = self.data_files[self.ids[self.file_idx]]
            label = np.array(self._load_file(os.path.join(self.image_name)))
            # Select only results which are not empty (e.g. only background)
            if(np.abs(np.max(label)-np.min(label))> 40):
                break

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
        myresult = cv2.resize(myresult, dsize=None, fx=.5, fy=.5)
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
        myresult = cv2.resize(myresult, dsize=None, fx=.5, fy=.5)
        myresult /= np.max(myresult)
        return myresult

