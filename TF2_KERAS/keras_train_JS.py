# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 08:58:10 2020

@author: diederichbenedict

Some sources:
    https://github.com/mnicholas2019/M202A/blob/master/PreviousWork/Activity-Recognition.ipynb
    https://github.com/vikranth94/Activity-Recognition/blob/master/existing_models.py

"""

import numpy as np
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from tensorflow.keras import optimizers

import keras_datagenerator as data
import keras_network as net

import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt

from datetime import datetime
from sys import platform
import io

import matplotlib as mpl
mpl.rc('figure',  figsize=(24, 20))


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


''' Some functions for displaying training progress
'''

def plot_to_image():
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.show()
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def log_images(epoch, logs):
    
    # Use the model to predict the values from the validation dataset.
    myX,myY = validation_generator.__getitem__(np.random.randint(100))
    myY_result = model.predict(myX)

    # Display
    myX_reshape = np.reshape(myX, (Nbatch, Ntime, Nx//2, Ny//2, 1))
    myY_reshape = np.reshape(myY, (Nbatch, Nx, Ny, 1))
    myY_result_reshape = np.reshape(myY_result, (Nbatch, Nx, Ny, 1))

    plt.figure()
    plt.subplot(221)
    plt.title('SOFI'), plt.imshow(np.squeeze(myY_result_reshape[0,]),cmap='gray')
    plt.subplot(222)
    plt.title('SR'), plt.imshow(np.squeeze(myY_reshape[0,]),cmap='gray')
    plt.subplot(223)
    plt.title('Mean'), plt.imshow(np.squeeze(np.mean(myX_reshape[0,],axis=(0,-1))),cmap='gray')
    plt.subplot(224)
    plt.title('STD'), plt.imshow(np.squeeze(np.std(myX_reshape[0,],axis=(0,-1))),cmap='gray')
    cm_image = plot_to_image()
      
    # Log the confusion matrix as an image summary.
    with file_writer_cm.as_default():
        tf.summary.image("Reconstructed Results", cm_image, step=epoch)


''' start the code here
'''
# Define Parameters
Ntime = 30
Nbatch = 1
Nfilterlstm = 16
Nx = 256    
Ny = 256
features = 1
Nchannel = 1
model_dir = "logs_bilinear/"# + datetime.now().strftime("%Y%m%d-%H%M%S")
network_type = 'SOFI_ConvLSTM2D'

# Training parameters
Nepochs = 150
Niter = 100


# Specify the location with for the training data #
if platform == "linux" or platform == "linux2":
    base_dir = './'
    data_dir = base_dir+'data_downconverted'; upscaling=2;
elif platform == "darwin":
	train_data_path = './test' # OS X
elif platform == 'win32':
    base_dir = ''#.\\'
    #data_dir = base_dir+'data\\\\data_downconverted_4'; upscaling=4;
    #data_dir = base_dir+'data\\\\data_raw'; upscaling=1;
    data_dir = base_dir+'C:\\Users\\diederichbenedict\\Dropbox\\Dokumente\\Promotion\\PROJECTS\\STORMoChip\\PYTHON\\Learn2Fluct\\convLSTM_predicttimeseries\data\\data'; upscaling=2;


# define the data generator
training_generator = data.DataGenerator(data_dir, n_batch=Nbatch,
                mysize=(Nx, Ny), n_time=Ntime, downscaling=upscaling, \
                n_modes = 40, mode_max_angle = 15, n_photons = 100, n_readnoise = 10, 
                kernelsize=2, quality_jpeg=80)
validation_generator = data.DataGenerator(data_dir, n_batch=Nbatch,
                mysize=(Nx, Ny), n_time=Ntime, downscaling=upscaling, \
                n_modes = 40, mode_max_angle = 15, n_photons = 100, n_readnoise = 10, 
                kernelsize=2, quality_jpeg=80)
        

# test dataloader 
i_testimage=17
myX,myY = training_generator.__getitem2D__(i_testimage)
# display images
plt.subplot(131)
plt.title('SR'), plt.imshow(np.squeeze(myY[0,])), plt.colorbar()
plt.subplot(132)
plt.title('mean'), plt.imshow(np.squeeze(np.mean(myX[0,],axis=(0,-1)))), plt.colorbar()
plt.subplot(133)
plt.title('raw'), plt.imshow(np.squeeze(myX[0,0,:,:,0])), plt.colorbar()
plt.show()

# create the model
print('Create the model!')
model = net.SOFI(Ntime=Ntime, Nbatch=Nbatch, Nx=Nx, Ny=Ny, features=features, Nchannel=Nchannel, Nfilterlstm=Nfilterlstm)
input_shape=(1,Nbatch*Nx//2*Ny//2*Ntime)
model.build(input_shape)

print(model.summary())



# compile the model 
print('Compile the model!')
#optimizer = optimizers.Nadam(lr=0.01, beta_1=0.9, beta_2=0.999)
optimizer = optimizers.Adam()
#losstype = tf.keras.losses.mean_squared_error
losstype = tf.keras.losses.mean_absolute_error
model.compile(loss=losstype,
              optimizer=optimizer,
              metrics=['accuracy'])

print(model.summary())

model.save('test.hdf5')


# Save model checkpoints # Define the per-epoch callback.
name = network_type+str(time.time())
tensorboard_callback = tensorboard = TensorBoard(log_dir=model_dir)
file_writer_cm = tf.summary.create_file_writer(model_dir)
cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_images)

# write out the logging files
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
filepath= f"best_{name}.hdf5"
chk_path = os.path.join(model_dir, filepath)
checkpoint = ModelCheckpoint(chk_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    steps_per_epoch = Niter, 
                    epochs=Nepochs,
                    callbacks=[tensorboard_callback,checkpoint,cm_callback], validation_steps=1)

# Predict 
batch_size_test=np.random.randint(0,300)
myX,myY = training_generator.__getitem__(batch_size_test)
myY_result = model.predict(myX, batch_size=Nbatch)

# Display
myX_reshape = np.reshape(myX, (Nbatch, Ntime, Nx//2, Ny//2, 1))
myY_reshape = np.reshape(myY, (Nbatch, Nx, Ny, 1))
myY_result_reshape = np.reshape(myY_result, (Nbatch, Nx, Ny, 1))

plt.figure()
plt.subplot(131)
plt.title('SOFI'), plt.imshow(np.squeeze(myY_result_reshape[0,])), plt.colorbar()
plt.subplot(132)
plt.title('SR'), plt.imshow(np.squeeze(myY_reshape[0,])), plt.colorbar()
plt.subplot(133)
plt.title('Mean'), plt.imshow(np.squeeze(np.mean(myX_reshape[0,],axis=(0,-1)))), plt.colorbar()
plt.show()
#


## THIS DOES NOT WORK WITH THE TF-GPU VERSION ON WINDOWS!!
model.save('test.hdf5')
# save model to drive
# tf.keras.models.save_model(
#     model = model,
#     filepath = 'test.hdf5',
#     overwrite=True,
#     include_optimizer=True,
#     save_format=None,
#     signatures=None
# )



################## TF JS 
#%% export tflite
import tensorflowjs as tfjs
import tensorflow as tf

model = tf.keras.models.load_model('test.hdf5')
model.summary()
Nbatch, Ntime, Nx, Ny = model.layers[1].output_shape

filepath= 'C://Users//diederichbenedict//Dropbox//Dokumente//Promotion//PROJECTS//STORMoChip//WEBSITE//stormocheap//STORMjs//'
filename = 'converted_model'+str(Nx)+'_'+str(Ntime)+'_keras'
tfjs.converters.save_keras_model(model, filepath+filename) 

# Bugfix
import json

with open(filepath+filename+'//model.json') as json_file:
    data = json.load(json_file)
    data['modelTopology']['model_config']['class_name']='Model'
    
with open(filepath+filename+'//model.json', 'w') as outfile:
    json.dump(data, outfile)