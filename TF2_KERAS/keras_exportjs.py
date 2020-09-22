#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 19:03:12 2020

@author: bene
"""
"""
Created on Thu Apr  2 08:58:10 2020

@author: 

Some sources:
    https://github.com/mnicholas2019/M202A/blob/master/PreviousWork/Activity-Recognition.ipynb
    https://github.com/vikranth94/Activity-Recognition/blob/master/existing_models.py
"""


#%% export tflite
import tensorflowjs as tfjs
import tensorflow as tf
import json
model = tf.keras.models.load_model('test.hdf5')
model.summary()
Nbatch, Ntime, Nx, Ny = model.layers[1].output_shape

filepath= 'C://Users//diederichbenedict//Dropbox//Dokumente//Promotion//PROJECTS//STORMoChip//WEBSITE//stormocheap//STORMjs//'
filename = 'converted_model'+str(Nx)+'_'+str(Ntime)+'_keras'
tfjs.converters.save_keras_model(model, filepath+filename) 

# Bugfix


with open(filepath+filename+'//model.json') as json_file:
    data = json.load(json_file)
    data['modelTopology']['model_config']['class_name']='Model'
    
with open(filepath+filename+'//model.json', 'w') as outfile:
    json.dump(data, outfile)