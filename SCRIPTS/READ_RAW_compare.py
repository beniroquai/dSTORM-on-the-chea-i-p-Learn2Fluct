# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 18:34:24 2020

@author: diederichbenedict
"""

import tifffile as tif 
import matplotlib.pyplot as plt
import rawpy
myfilename = 'C:\\Users\\diederichbenedict\\Dropbox\\TMP\\2020_06_12-dSTORM_Raw_vs_MP4\\RAW_dSTORM_3_full\\2020_06_12_17_04_23_59.dng'
myimage = tif.imread(myfilename)
myimage_raw = rawpy.imread(myfilename) 
myimage_raw = myimage_raw.raw_image# ()#gamma=(1,1), no_auto_bright=True, output_bps=16)
plt.subplot(131), plt.title('TIFFFILE'), plt.imshow(myimage, vmax=np.mean(myimage)*2, cmap='gray')
plt.subplot(132), plt.title('PYRAW'), plt.imshow(myimage_raw, vmax=np.mean(myimage_raw)*2, cmap='gray')
plt.subplot(133), plt.title('DIFF'), plt.imshow(myimage-myimage_raw, cmap='gray')

