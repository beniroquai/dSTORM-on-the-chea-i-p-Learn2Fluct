#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:06:16 2020

@author: bene
"""

import glob
import cv2
import numpy as np
import NanoImagingPack as nip
import tifffile as tif
import matplotlib.pyplot as plt

myfolder = '/Users/bene/Dropbox/20200318_121415884/'
myfolder = '/Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/STORMoChip/DATA/20200422_181103263_EColi_Fluct/'
myfolder = '/Users/bene/Downloads/20200504_084015766__EColi_Fluct_stability/'
myfolder = '/Users/bene/Downloads/20200505_170647713_HUVEC_Phaloidin_AF647/'
# myfolder = '/Users/bene/Downloads/20200505_160633020_HUVECS_SYTO6/'
myframes = []
tperiod = 1
mytime = 0

roi = 1024

for ifile in range(1,100):
    try:
        filename=myfolder+'VID_'+str(ifile)+'.mp4'
        cap = cv2.VideoCapture(filename)
        print(filename)
        gray = []
        for iframe in range(10):
            # Capture frame-by-frame
            ret, frame = cap.read()
        
            # Our operations on the frame come here
            gray.append(np.float32(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
        gray = np.mean(np.array(gray), axis=0)
        gray -= np.min(gray)        
        gray = nip.extract(gray, (roi,roi))
        mymean = np.mean(gray)
        gray/=mymean
        
        #plt.imshow(cv2.putText(img=np.copy(gray), text="t="+str(mytime)+" min", org=(roi//2+200, roi//2+400),fontFace=2, fontScale=1, color=(1,1,1), thickness=2))
        gray = (cv2.putText(img=np.copy(gray), text="t="+str(mytime)+" min", org=(roi//2+200, roi//2+400),fontFace=2, fontScale=1, color=(int(mymean),int(mymean),int(mymean)), thickness=2))
        #gray =cv2.putText(img=np.copy(gray), text="t="+str(mytime)+" min", org=(850-roi,1000-roi),fontFace=2, fontScale=1, color=(1,1,1), thickness=2)
        mytime += tperiod

        myframes.append(gray)
    except:
        print('Could not read video...')

    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
myframes = np.array(myframes)
myframes = myframes/np.expand_dims(np.expand_dims(np.max(myframes, axis=(1,2)), -1),-1)
# myframes  DONT DO THAT TIME embossed = myframes-np.mean(myframes,axis=0)

# save images 
tif.imsave(myfolder+'myresults.tif', myframes, append=False)

   # do your stuff