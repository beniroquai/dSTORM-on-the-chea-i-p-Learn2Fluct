# -*- coding: utf-8 -*-
import skvideo.io
import skvideo.datasets
import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt
import NanoImagingPack as nip 
import cv2

from skimage.feature import register_translation
from skimage.transform import AffineTransform



def find_shift_sub(im1, im2):
    # subpixel precision
    # pixel precision first
    shift, error, diffphase = register_translation(im1, im2, 150)
    #print("Detected subpixel offset (y, x): {}".format(shift))
    return shift    

#myvideofile = '2019-06-13_15.07.46_Fluctuation_Cardio_Mayoblast_Cellphone_Largelens.mp4'

myvideopath = 'C://Users//diederichbenedict//Dropbox//STORMonAcheaip//STORM-on-a-chea(i)p//'
myvideofile = 'MOV_2019_07_09_17_24_52.mp4'
myvideofile = 'MOV_2019_07_09_17_24_24.mp4'

myvideopath = 'C://Users//diederichbenedict//Dropbox//Camera Uploads//'

myvideofile = '2019-08-06 16.57.35.mp4'
myvideofile = '2019-08-06 16.46.29.mp4'

#myvideopath = 'C:/Users/diederichbenedict/Downloads/'
myvideofile = 'MOV_2019_08_12_10_41_31.mp4'

myvideofile = '2019-08-16 12.55.06.mp4'
myvideopath = './'
myvideofile = 'MOV_2019_09_02_19_07_46.mp4'
#myvideopath = '/Users/bene/Downloads/'
#myvideofile = '2019-06-20 18.41.00_blink.mp4'

# cool example for mode scanning

myvideopath = 'C://Users//diederichbenedict//Downloads//'
myvideofile = '2019-09-03 15.51.12.mp4'
myvideofile = '2019-10-15 06.46.57.mp4'
myvideofile = '2019-10-15 17.39.17.mp4'
myvideofile = '2019-10-16 06.41.19.mp4'
myvideofile = '2019-10-17 06.53.08.mp4' # 60xobjective
myvideofile = '2019-10-18 11.37.14.mp4'
myvideofile = '2019-10-19 07.48.52.mp4'
myvideofile = 'MOV_2019_10_20_08_54_54.mp4'
myvideofile = 'MOV_2019_10_21_07_27_51.mp4'
myvideofile = 'MOV_2019_10_23_09_17_18.mp4'; mycenterpos = (1260,1840)

outputfile = myvideofile+'.tif'
myvideofile = myvideopath+myvideofile
videogen = skvideo.io.vreader(myvideofile)

myroisize = 512 #2048
iiter = 0
myimlist = []
myshiftlist = []
for frame in videogen:
    gray = np.mean(frame, 2)#frame[:,:,1] # R,G,B = 0,1,2
    gray = nip.extract(gray, ROIsize=(myroisize,myroisize), centerpos=mycenterpos)

    #gray = cv2.resize(gray,(int(myroisize//2),int(myroisize//2)))
    #nip.view(gray)
    #tif.imsave(outputfile, np.uint8(gray), append=True, bigtiff=True) #compression='lzw',     
    iiter+=1
    print(iiter)
    myimlist.append(gray)


myimage = np.array(myimlist)
#myimage *= (myimage>17)

tif.imsave(myvideopath+outputfile, np.uint8(myimage), append=False, bigtiff=True) #compression='lzw',     

adsf
#%% Estimate the shift of the stack over time
# - > some kind of fiducial marker thingy
from scipy.ndimage import shift
from scipy import ndimage
myframelist = []
myroilist = []
myroishiftlist = []

mysubroisize = 200
mycenter = (mysubroisize//2, mysubroisize//2)
myfeature = (435,226)#(1205,1657)

from scipy.ndimage.filters import gaussian_filter
for jiter in range(0, myimage.shape[0]):
    myframe = myimage[jiter,:,:]
    gray_sub = nip.extract(myframe, ROIsize=(mysubroisize,mysubroisize), centerpos=myfeature )

    gray_sub_blurred = gaussian_filter(np.uint8(np.array(np.squeeze(gray_sub))), sigma=2)
    mymaxpos = np.unravel_index(gray_sub_blurred.argmax(), gray_sub_blurred.shape)
    
    myshift_x = mycenter


    mydiffshift = (mycenter-np.array(mymaxpos))
    myframe = shift(myframe, (mydiffshift[0], mydiffshift[1]))
    gray_sub_blurred = shift(gray_sub_blurred, (mydiffshift[0], mydiffshift[1]))

        

    myframelist.append(myframe)
    myroilist.append(gray_sub)
    myroishiftlist.append(gray_sub_blurred)
    
    print("My Shift @ "+str(jiter)+" is: "+str(mydiffshift))

myimage_shifted = np.array(myframelist)
tif.imsave(outputfile, np.uint8(myimage_shifted), append=False, bigtiff=True) #compression='lzw',     
