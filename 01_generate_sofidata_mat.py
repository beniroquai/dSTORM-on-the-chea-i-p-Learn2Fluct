import os
import requests
import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt
from io import BytesIO
import cv2
import NanoImagingPack as nip
import scipy.io as sio
from skimage.draw import line_aa
from scipy.ndimage import gaussian_filter

# This script automatically downloads a bunch of images from the OMERO web viwer
# https://idr.openmicroscopy.org/mapr/cellline/?experimenter=-1
image_id_min = 4995167, which_channel = 'R'# some actin stained U2OS cells 
image_id_min  = 9757045 # some huvecs
image_id_min = 9757045, which_channel = 'G' # some U2OS
num_images = 100
myfolder = './convLSTM_predicttimeseries/data_downconverted'
myfolder_raw = myfolder + './'

is_safemat = True   # want to store the data as mats?

n_modes = 30
n_frames = 300
gaussianstd= 1
n_photons = 100
n_readnoise = 10
quality_jpeg = 70
downscaling = 2
mode_max_angle = 15 # degrees
mysize_new = (200,200)  # size for the dataset 



try: os.mkdir(myfolder)
except: print('Folder exists')

try: os.mkdir(myfolder_raw)
except: print('Folder exists')

        
for i_images in range(num_images):

    # download the image
    myimagefile = str(image_id_min+i_images)
    print('Downloading Image: '+myimagefile+', '+str(i_images)+'/'+str(num_images))
    url = 'https://idr.openmicroscopy.org/webclient/render_image_download/'+myimagefile+'/?format=tif'
    page = requests.get(url)
    
    # odd for now, but don't know how to cast TIFF to numpy without decoding using a file
    f_name = myfolder_raw + '/' + str(myimagefile)+'.tif'
    with open(f_name, 'wb') as f:
        f.write(page.content)
    myimagetmp = tif.imread(f_name)
    
    # split image and select the channel of choisce
    if which_cannel=='R':
            mysample = myimagetmp[:,:,0]
    elif which_cannel=='G':
            mysample = myimagetmp[:,:,1]
    elif which_cannel=='B':
            mysample = myimagetmp[:,:,2]            
            
    # ---------------------------------------------------
    # produce some SOFI data for the on-chip simulation
    # ---------------------------------------------------

    # normalize the sample
    mysample = nip.extract(nip.resample(mysample, factors =(.25,.25)), mysize_new)
    mysample -= np.min(mysample)
    mysample = mysample/np.max(mysample)*(2**8-1)

    # allocate some memory
    myallframes_illu = []
    myallframes_obj = []
    # iterate over all frames
    print('Illuminating the frames')
    for i_frame in range(n_frames):
        myallmodes = np.zeros(mysize_new)*255
        
        # generate illuminating modes
        for i_modes in range(n_modes):
            while(True):
                start_x = np.random.randint(0,mysize_new[0])
                start_y = np.random.randint(0,mysize_new[1])
                # make sure the angle is not too steep!
                if(np.abs(np.arctan((start_x-start_y)/mysize_new[0])/np.pi*180)<mode_max_angle):
                    break
            # https://stackoverflow.com/questions/31638651/how-can-i-draw-lines-into-numpy-arrays            
            rows, cols, weights = line_aa(start_x, 0, start_y, mysize_new[0]-1)    # antialias line
            w = weights.reshape([-1, 1])
            lineColorRgb = [1, 1, 1]  
            #np.multiply((1 - w),myallmodes[rows, cols]) + w * np.array([lineColorRgb])
            myallmodes[rows, cols] = np.random.randint(50,100)/100
        # print('Frame: '+str(i_frame))

        
        # illuminate the sample with the structured illumination 
        myresultframe = gaussian_filter(myallmodes*mysample, sigma=gaussianstd)
        
        # add noise
        myresultframe = nip.noise.poisson(myresultframe, n_photons)
        myresultframe = myresultframe + n_readnoise*np.random.randn(myresultframe.shape[0],myresultframe.shape[1])
        
        # subsample data
        myresultframe = nip.resample(myresultframe, [1/downscaling, 1/downscaling])
        
        # add compression artifacts 
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality_jpeg]
        result, encimg = cv2.imencode('.jpg', myresultframe, encode_param)
        myresultframe_compressed = np.mean(cv2.imdecode(encimg, 1),-1)
        
    
        # save all images
        myallframes_obj.append(myresultframe_compressed)
        myallframes_illu.append(myallmodes)
    
    # save results
    myfolderdataset = myfolder + '/' + str(myimagefile) + '/'
    try:
        os.mkdir(myfolderdataset)
    except:
        print('Folder exists')
    
    print('Savina ll frames')
    myallframes_illu = np.float32(np.array(myallframes_illu))
    myallframes_obj = np.transpose(np.uint8(np.array(myallframes_obj)), [1,2,0])
    
    if(is_safemat):
        sio.savemat(myfolderdataset+'sofi.mat', {'sofi':myallframes_obj})
        sio.savemat(myfolderdataset+'sr.mat', {'sr':np.uint8(mysample)})
    else:
        tif.imsave(myfolderdataset+'mytest.tif', myallframes)
        tif.imsave(myfolderdataset+'mytest_obj.tif', myallframes_obj)
        tif.imsave(myfolderdataset+'mytest_gt.tif', np.float32(mysample))
    
    # show result
    plt.subplot(131)
    plt.imshow(myallframes_obj[:,:,0]), plt.colorbar()
    plt.subplot(132)
    plt.imshow(myallframes_illu[0,:,:]), plt.colorbar()
    plt.subplot(133)
    plt.imshow(mysample), plt.colorbar()
    plt.show()
