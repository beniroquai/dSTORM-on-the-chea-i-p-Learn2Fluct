import sofilstm.sofi as sofi
gt = np.float32(np.array(gt_mat['sr']))import sofilstm.train as train
from sofilstm import image_util

import scipy.io as spio
import numpy as np
import os
import NanoImagingPack as nip
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
mpl.rc('figure',  figsize=(12, 9))
mpl.rc('image', cmap='gray')

####################################################
####             PREPARE WORKSPACE               ###
####################################################


# Because we have real & imaginary part of our input, data_channels is set to 2
data_channels = 1
truth_channels = 1

# here indicating the GPU you want to use. if you don't have GPU, just leave it.
gpu_ind = '-1'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ind; # 0,1,2,3

# Because we have real & imaginary part of our input, data_channels is set to 2
data_channels = 1
truth_channels = 1
features_root = 1
# args for training
batch_size = 1     # batch size for training
ntimesteps = 30    # number of time-steps for one 3D volume
optimizer = "adam"  # optimizer we want to use, 'adam' or 'momentum'
Nx, Ny = 128, 128 # final size of the image

learning_rate = 0.01
display_step = 50
epochs = 300
training_iters = 200
save_epoch = 1



# here specify the path of the model you want to load
model_path = './final/'#'./networks/upsamping_2_lstm4_128x128_time_30_batchnorm_newdataprovider_subpixel_gpu0/models/final/'
####################################################
####                 FUNCTIONS                   ###
####################################################

# make the data a 4D vector
def preprocess(data, mysize=None):
    if ~(mysize is None):
        data = nip.extract(data, mysize)
    data -= np.min(data)
    data = data/np.max(data)
    return data

####################################################
####                lOAD MODEL                   ###
####################################################

net = sofi.SOFI(batchsize=batch_size, Nx=Nx, Ny=Ny, img_channels=1, features_root=features_root, ntimesteps=ntimesteps, cost="mean_squared_error")

#
#%%
path_mydata = './test/sofi.mat'
path_mydata = './test/sofi_ecoli.mat'
path_mydata = './test/4995169/'

#preparing training data
from skimage.transform import resize
data_mat = spio.loadmat(path_mydata + 'sofi.mat', squeeze_me=True)
gt_mat = spio.loadmat(path_mydata + 'sr.mat', squeeze_me=True)

upscaling = 1
data = np.float32(np.array(data_mat['sofi']))
gt = np.float32(np.array(gt_mat['sr']))
Nx,Ny,Nz = data.shape[0]*2,data.shape[1]*2,data.shape[2]
#data = resize(data, (Nx*upscaling, Nx*upscaling, Nz))
data = preprocess(data, mysize=(Nx//2,Ny//2,ntimesteps))
data = np.expand_dims(data,0)**.52
#data = np.concatenate((data,data), 0)





####################################################
####              	  PREDICT                    ###
####################################################
# save_path = net.savegraph(model_path, model_path)
# # convert model
# # Converting a SavedModel to a TensorFlow Lite model.
# import tensorflow.lite as lite
# converter = lite.TFLiteConverter.from_saved_model(save_path)
# tflite_model = converter.convert()

#net.saveTFLITE(model_path+'model.cpkt')

predict = net.predict(model_path+'model.cpkt', data, 1, True)
#predict(self, model_path, x_test, keep_prob, phase)

#% visualize 

plt.figure()
plt.subplot(321), plt.title('prediction')
plt.imshow(np.squeeze(predict[0,:,:,0])), plt.colorbar()
plt.subplot(323), plt.title('std')
plt.imshow(np.std(data[0,:,:,:],-1)), plt.colorbar()
plt.subplot(324), plt.title('mean')
plt.imshow(np.mean(data[0,:,:,:],-1)), plt.colorbar()
plt.subplot(322), plt.title('raw')
plt.imshow(data[0,:,:,0]), plt.colorbar()
plt.subplot(325), plt.title('groundtruth')
plt.imshow(gt), plt.colorbar()
plt.subplot(326), plt.title('groundtruth')
plt.imshow(gt), plt.colorbar()

# plt.figure()
# plt.subplot(211), plt.title('prediction')
# plt.imshow(np.std(data[0,:,:,:],-1)), plt.colorbar()
# #plt.imshow(np.squeeze(predict[0,:,:,0])), plt.colorbar()
# plt.subplot(212), plt.title('groundtruth')
# plt.imshow(gt), plt.colorbar()