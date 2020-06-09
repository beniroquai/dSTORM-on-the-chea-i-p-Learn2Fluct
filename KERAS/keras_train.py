# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 08:58:10 2020

@author: diederichbenedict

Some sources:
    https://github.com/mnicholas2019/M202A/blob/master/PreviousWork/Activity-Recognition.ipynb
    https://github.com/vikranth94/Activity-Recognition/blob/master/existing_models.py
"""

import numpy as np

from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model
from keras import optimizers

import keras_datagenerator as data
import keras_network as net
import keras_utils as utils
import keras_tensorboard as ktb

import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from sys import platform
# Define Parameters
Ntime = 30
Nbatch = 1
Nx = 100
Ny = 100
features = 1
Nchannel = 1
model_dir = '.\\\\network'
network_type = 'SOFI_ConvLSTM2D'

Nepochs = 1
Niter = 100
Ndisplay = 20

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
list_IDs = []
labels = []
training_generator = data.DataGenerator(batch_size=Nbatch, dim=(Nx,Ny), data_dir=data_dir, time_steps=Ntime, n_channels=Nchannel, shuffle=True, upscaling=upscaling)
validation_generator = data.DataGenerator(batch_size=Nbatch, dim=(Nx,Ny), data_dir=data_dir, time_steps=Ntime, n_channels=Nchannel, shuffle=True, upscaling=upscaling)

# test dataloader 
i_testimage=17
myX,myY = training_generator.__getitem__(i_testimage)
# display images
plt.subplot(131)
plt.title('SR'), plt.imshow(np.squeeze(myY)), plt.colorbar()
plt.subplot(132)
plt.title('mean'), plt.imshow(np.squeeze(np.mean(myX,axis=1))), plt.colorbar()
plt.subplot(133)
plt.title('raw'), plt.imshow(np.squeeze(myX[:,1,:,:,:])), plt.colorbar()
plt.show()

# create the model
print('Create the model!')
model = net.SOFI(Ntime=Ntime, Nbatch=Nbatch, Nx=Nx, Ny=Ny, features=features, Nchannel=Nchannel)
print(model.summary())

# add summaries
tf.summary.image('output_image', utils.get_image_summary(model.output))
tf.summary.image('input_image_mean', utils.get_image_summary(tf.reduce_mean(model.input,1)))
tf.summary.image('input_image_raw', utils.get_image_summary(model.input[:,1,:,:,:]))


# compile the model 
print('Compile the model!')
#optimizer = optimizers.Nadam(lr=0.01, beta_1=0.9, beta_2=0.999)
optimizer = optimizers.adam()
#losstype = tf.keras.losses.mean_squared_error
losstype = tf.keras.losses.mean_absolute_error
model.compile(loss=losstype,
              optimizer=optimizer,
              metrics=['accuracy'])

# Save model checkpoints
name = network_type+str(time.time())
tensorboard_logdir= 'logs/{}'.format(name)
print("Logdir of Tensorflow is: "+tensorboard_logdir)
tensorboard = TensorBoard(log_dir = tensorboard_logdir,update_freq=Ndisplay)
# initialize the variables and the `tf.assign` ops
cbk = ktb.CollectOutputAndTarget(tensorboard_logdir)
fetches = [tf.assign(cbk.var_y_true, model.targets[0], validate_shape=False),
           tf.assign(cbk.var_y_pred, model.outputs[0], validate_shape=False),
           tf.assign(cbk.var_x_true, model.inputs[0], validate_shape=False),]
model._function_kwargs = {'fetches': fetches}  # use `model._function_kwargs` if using `Model` instead of `Sequential`



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
                    steps_per_epoch = Niter, epochs=Nepochs,
                    callbacks=[tensorboard, checkpoint,cbk], validation_steps=1)#,                    workers=2)


model.save(f'final_{name}.hdf5')

# Reactivate model
#model = load_model(chk_path)

# Predict 
batch_size_test=np.random.randint(0,300)
myX,myY = training_generator.__getitem__(batch_size_test)
myY_result = model.predict(myX, batch_size=Nbatch)

plt.subplot(131)
plt.title('SOFI'), plt.imshow(np.squeeze(myY_result)), plt.colorbar()
plt.subplot(132)
plt.title('SR'), plt.imshow(np.squeeze(myY)), plt.colorbar()
plt.subplot(133)
plt.title('Mean'), plt.imshow(np.squeeze(np.mean(myX,axis=1))), plt.colorbar()
plt.show()
#


# safe for tflite
from tensorflow import lite
import tensorflow as tf
saved_model = ''
converter = lite.TFLiteConverter.from_keras_model_file(saved_model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
open(os.path.join(model_dir,"converted_model.tflite"), "wb").write(tflite_quant_model)


import tensorflow as tf
MODEL_PATH = f'final_{name}.hdf5'
converter = tf.lite.TFLiteConverter.from_keras_model_file(MODEL_PATH)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]

tflite_model = converter.convert()
open("test.tflite", "wb").write(tflite_model)