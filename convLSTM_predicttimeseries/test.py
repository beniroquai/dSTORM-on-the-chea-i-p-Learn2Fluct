import sofilstm.sofi as sofi
import sofilstm.train as train
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
gpu_ind = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ind; # 0,1,2,3


# here specify the path of the model you want to load
model_path = './networks/upsamping_4_noconv_100x100_2_gpu0/models/final/'


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

# set up args for the unet, should be exactly the same as the loading model
kwargs = {
    "layers": 5,
    "conv_times": 2,
    "features_root": 64,
    "filter_size": 3,
    "pool_size": 2,
    "summaries": True
}

# args for training
batch_size = 1  # batch size for training
ntimesteps = 1  # number of time-steps for one 3D volume
valid_size = batch_size  # batch size for validating (must be same as batch_size!)
Nx, Ny = 100, 100

path_mydata = './test/sofi.mat'
path_mydata = './test/sofi_ecoli.mat'

#preparing training data
from skimage.transform import resize
data_mat = spio.loadmat(path_mydata , squeeze_me=True)
upscaling = 4
data = np.float32(np.array(data_mat['sofi']))
Nx,Ny,Nz = data.shape[0],data.shape[1],data.shape[2]
data = resize(data, (Nx*upscaling, Nx*upscaling, Nz))
data = preprocess(data, mysize=(Nx*upscaling,Ny*upscaling,ntimesteps))
data = np.expand_dims(data,0)
#data = np.concatenate((data,data), 0)



net = sofi.SOFI(batchsize=batch_size, Nx=Nx*upscaling, Ny=Ny*upscaling, ntimesteps=ntimesteps, cost="mean_squared_error", **kwargs)


####################################################
####              	  PREDICT                    ###
####################################################
# save_path = net.savegraph(model_path, model_path)
# # convert model
# # Converting a SavedModel to a TensorFlow Lite model.
# import tensorflow.lite as lite
# converter = lite.TFLiteConverter.from_saved_model(save_path)
# tflite_model = converter.convert()

net.saveTFLITE(model_path+'model.cpkt')

predict = net.predict(model_path+'model.cpkt', data, 1, True)
#predict(self, model_path, x_test, keep_prob, phase)

#%% visualize 

plt.figure()
plt.subplot(321), plt.title('prediction')
plt.imshow(np.squeeze(predict[0,:,:,0])), plt.colorbar()
plt.subplot(323), plt.title('std')
plt.imshow(np.std(data[0,:,:,:],-1)), plt.colorbar()
plt.subplot(324), plt.title('mean')
plt.imshow(np.mean(data[0,:,:,:],-1)), plt.colorbar()
plt.subplot(322), plt.title('raw')
plt.imshow(data[0,:,:,0]), plt.colorbar()
plt.subplot(325), plt.title('mean-prediction')
plt.imshow(np.log(1+np.array(predict[0,:,:,0]-np.mean(data[0,:,:,:],-1)))), plt.colorbar()