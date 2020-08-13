# -*- coding: utf-8 -*-

import rawpy
import imageio
import matplotlib.pyplot as plt 
import numpy as np
import NanoImagingPack as nip
import glob
import tifffile as tif
import os

#%% Read RAW-images and fuse them to a tif
mypath = './DNGs/'
myextension = '*.dng'

myimage_list = []
myimage_filelst = sorted(glob.glob(mypath+myextension), key=os.path.getmtime)

for ifile in range(0,len(myimage_filelst)):
    myfilename = myimage_filelst[ifile]
    with rawpy.imread(myfilename) as raw:
        print('Reading file: ' + str(myfilename))
        myimage_raw = raw.raw_image# ()#gamma=(1,1), no_auto_bright=True, output_bps=16)
        myimage = np.array(myimage_raw)
        myimage_crop = nip.extract(myimage, (myimage.shape[0]-4, myimage.shape[1]-4)) # need to crop due to 0-pixels in tif-writer
        myimage_list.append(myimage_crop)
        

myimages = np.asarray(myimage_list)
tif.imsave('my_raw_result.tif', myimages, dtype='uint16', append=False)



with rawpy.imread(myfilename) as raw:
        myimage_raw = raw.raw_image# ()#gamma=(1,1), no_auto_bright=True, output_bps=16)
        myimage = np.array(myimage_raw)
        myimage_crop = nip.extract(myimage, (myimage.shape[0]-4, myimage.shape[1]-4)) # need to crop due to 0-pixels in tif-writer
        myimage_list.append(myimage_crop)
