# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:50:54 2020

@author: diederichbenedict
"""
import tensorflow as tf
import numpy as np

from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model
from keras import optimizers

import keras_datagenerator as data
import keras_network as net
import keras_utils as utils
import keras_tensorboard as ktb

import os
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sys import platform
import io

def get_image_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """
    
    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255
    
    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V


def norm_image(img):
    img -= np.min(img)
    img /= np.max(img)
    return img
    
    
