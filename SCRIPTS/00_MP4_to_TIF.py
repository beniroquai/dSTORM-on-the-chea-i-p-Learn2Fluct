# -*- coding: utf-8 -*-
import skvideo.io
import skvideo.datasets
import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt
import NanoImagingPack as nip 


myvideopath = '/Users/Bene/Downloads/'
myvideofile = '2019-09-19 10.47.23.mp4'
mycenterpos=(500,1850)
myroisize = 256


myvideofile = '2019-09-19 08.12.11.mp4'
mycenterpos=(1000,1500)


myvideofile = '2019-09-19 07.57.13.mp4'
mycenterpos=(1140,1750)
myroisize = 512


outputfile = myvideofile+'.tif'
myvideofile = myvideopath+myvideofile
videogen = skvideo.io.vreader(myvideofile)


iiter = 0
myimlist = []
for frame in videogen:
    #%%
    gray = np.mean(frame, 2)

    gray = nip.extract(gray, ROIsize=(myroisize,myroisize), centerpos=mycenterpos)
#    nip.view(gray)
    
    
    #%%
    #tif.imsave(outputfile, np.uint8(gray), append=True, bigtiff=True) #compression='lzw',     
    
    iiter+=1
    #kljh
    
    print(iiter)
    myimlist.append(gray)

myimage = np.array(myimlist[230:-1])
tif.imsave(outputfile, np.uint8(myimage), append=False, bigtiff=True) #compression='lzw',     

