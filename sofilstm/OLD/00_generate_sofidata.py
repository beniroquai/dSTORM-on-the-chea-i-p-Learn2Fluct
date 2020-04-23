import os
import requests
import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt
from io import BytesIO
import cv2
import NanoImagingPack as nip

# This script automatically downloads a bunch of images from the OMERO web viwer
image_id_min = 4995167
num_images = 100
myfolder = './OMERO'
myfolder_r = './OMERO/R'
myfolder_g = './OMERO/G'
myfolder_b = './OMERO/B'


try:
    os.mkdir(myfolder_r)
    os.mkdir(myfolder_g)
    os.mkdir(myfolder_b)
    os.mkdir(myfolder)
except:
    print('Folder exists')

for i_images in range(num_images):

    myimagefile = str(image_id_min+i_images)
    print('Downloading Image: '+myimagefile+', '+str(i_images)+'/'+str(num_images))
    url = 'https://idr.openmicroscopy.org/webclient/render_image_download/'+myimagefile+'/?format=tif'
    page = requests.get(url)
    
    
    # odd for now, but don't know how to cast TIFF to numpy without decoding using a file
    f_name = myfolder + '/tmp.tif'
    with open(f_name, 'wb') as f:
        f.write(page.content)
    myimagetmp = tif.imread(f_name)
    
    # split image 
    R,G,B = myimagetmp[:,:,0], myimagetmp[:,:,1], myimagetmp[:,:,2]

    f_name_r = myfolder_r + '/' + myimagefile+'_R.tif'
    f_name_g = myfolder_g + '/' + myimagefile+'_G.tif'
    f_name_b = myfolder_b + '/' + myimagefile+'_B.tif'
    
    # save images
    tif.imwrite(f_name_r, R)
    tif.imwrite(f_name_g, G)
    tif.imwrite(f_name_b, B)
    

# produce some SOFI data for the on-chip simulation
#%%
from skimage.draw import line_aa
from scipy.ndimage import gaussian_filter
n_modes = 100
n_frames = 100
gaussianstd= 4
n_photons = 100
mysize = G.shape
n_readnoise = 10
quality_jpeg = 50
mode_max_angle = 15 # degrees

myallframes_illu = []
myallframes_obj = []

# normalize the sample
mysample = R/np.max(R)

for i_frame in range(n_frames):
    myallmodes = np.zeros(mysize)*255
    for i_modes in range(n_modes):
        while(True):
            start_x = np.random.randint(0,mysize[0])
            start_y = np.random.randint(0,mysize[1])
            # make sure the angle is not too steep!
            if(np.abs(np.arctan((start_x-start_y)/mysize[0])/np.pi*180)<mode_max_angle):
                break
        # https://stackoverflow.com/questions/31638651/how-can-i-draw-lines-into-numpy-arrays            
        rows, cols, weights = line_aa(start_x, 0, start_y, mysize[0]-1)    # antialias line
        w = weights.reshape([-1, 1])
        lineColorRgb = [1, 1, 1]  
        #np.multiply((1 - w),myallmodes[rows, cols]) + w * np.array([lineColorRgb])
        myallmodes[rows, cols] = np.random.randint(50,100)/100
    print('Frame: '+str(i_frame))

    
    # illuminate the sample with the structured illumination 
    myresultframe = gaussian_filter(myallmodes*mysample, sigma=gaussianstd)
    
    # add noise
    myresultframe = nip.noise.poisson(myresultframe, 100)
    myresultframe = myresultframe + n_readnoise*np.random.randn(myresultframe.shape[0],myresultframe.shape[1])
    
    # add compression artifacts 
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality_jpeg]
    result, encimg = cv2.imencode('.jpg', myresultframe, encode_param)
    myresultframe_compressed = cv2.imdecode(encimg, 1)

    # save all images
    myallframes_obj.append(myresultframe_compressed)
    myallframes_illu.append(myallmodes)

# save results
myallframes_illu = np.float32(np.array(myallframes_illu))
myallframes_obj = np.uint8(np.array(myallframes_obj))
tif.imsave('mytest.tif', myallframes)
tif.imsave('mytest_obj.tif', myallframes_obj)
tif.imsave('mytest_gt.tif', np.float32(mysample))

# show result
plt.subplot(121)
plt.imshow(myallframes_obj[0,:,:]), plt.colorbar()
plt.subplot(122)
plt.imshow(myallframes_illu[0,:,:]), plt.colorbar()
plt.show()
