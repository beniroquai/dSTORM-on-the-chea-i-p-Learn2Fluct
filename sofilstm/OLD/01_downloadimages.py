import os
import requests
import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt


import cv2
import NanoImagingPack as nip
import scipy.io as sio
from skimage.draw import line_aa
from scipy.ndimage import gaussian_filter

    
# This script automatically downloads a bunch of images from the OMERO web viwer
# https://idr.openmicroscopy.org/mapr/cellline/?experimenter=-1
idataset = 1

mysize_new = [400,400]
if(idataset==1):
    image_id_min = 4995167+257
    which_channel = 'R'# some actin stained U2OS cells 
    downsampling = .25    
elif(idataset==2):
# some more actin vimentim HUVEC
    image_id_min = 4993676
    which_channel = 'G'
    downsampling = .25
elif(idataset==3):
# some more actin vimentim HUVEC
    image_id_min = 4990951
    downsampling = .25
    which_channel = 'G'
elif(idataset==4):
# some more actin vimentim HUVEC
    image_id_min = 4991076
    which_channel = 'G'
    downsampling = .25
elif(idataset==5):
    image_id_min  = 9757045 # some huvecs
    which_channel = 'G' # some U2OS
    downsampling = .8


num_images = 100

is_safemat = True   # want to store the data as mats?

myfolder = './convLSTM_predicttimeseries/data/data'
myfolder_raw = myfolder + './'

try: os.mkdir(myfolder)
except: print('Folder exists')

try: os.mkdir(myfolder_raw)
except: print('Folder exists')

        
for i_images in range(num_images):

    # download the image
    myimagefile = str(image_id_min+i_images)
    print('Downloading Image: '+myimagefile+', '+str(i_images)+'/'+str(num_images))
    url = 'http://idr.openmicroscopy.org/webclient/render_image_download/'+myimagefile+'/?format=tif'
    page = requests.get(url)
    # odd for now, but don't know how to cast TIFF to numpy without decoding using a file
    f_name = myfolder_raw + '/' + str(myimagefile)+'.tif'
    with open(f_name, 'wb') as f:
        f.write(page.content)    
    try:
        myimagetmp = tif.imread(f_name)
    except:
        print('Not an image!')
    
    
    # split image and select the channel of choisce
    if which_channel=='R':
            mysample = myimagetmp[:,:,0]
    elif which_channel=='G':
            mysample = myimagetmp[:,:,1]
    elif which_channel=='B':
            mysample = myimagetmp[:,:,2]            
            
    # ---------------------------------------------------
    # produce some SOFI data for the on-chip simulation
    # ---------------------------------------------------

    # normalize the sample
    mysample = nip.resample(mysample, factors =(downsampling,downsampling))
    mysample = nip.extract(mysample, mysize_new)
    mysample -= np.min(mysample)
    mysample = mysample/np.max(mysample)*(2**8-1)
    tif.imsave(f_name,mysample)
    plt.imshow(mysample), plt.colorbar()
    plt.show()
        

