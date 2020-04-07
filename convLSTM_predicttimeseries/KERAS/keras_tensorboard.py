# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:30:06 2020

@author: diederichbenedict
https://stackoverflow.com/questions/43784921/how-to-display-custom-images-in-tensorboard-using-keras?noredirect=1#comment85726690_43784921
"""

import tensorflow as tf
import keras 
import numpy as np
from keras.callbacks import Callback
from keras import backend as K
import matplotlib.pyplot as plt
import keras_utils as utils

def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    from PIL import Image
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)


class TensorBoardImage(Callback):
    def __init__(self, tag):
        super().__init__()
        self.tag = tag

    def on_batch_end(self, batch, logs={}):
        # Load image
        print( self.model.input[0].shape)  
        print( self.model.output[1].shape)  
        
        img_input = self.model.input[0]  # X_train
        img_valid = self.model.output[0]  # Y_train

        image = make_image(img_input)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag, image=image)])
        writer = tf.summary.FileWriter('./logs')
        writer.add_summary(summary, batch)
        writer.close()

        image = make_image(img_valid)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag, image=image)])
        writer = tf.summary.FileWriter('./logs')
        writer.add_summary(summary, batch)
        writer.close()

        return
    
class CollectOutputAndTarget(Callback):
    def __init__(self, path=''):
        super(CollectOutputAndTarget, self).__init__()
        self.targets = 0.  # collect y_true batches
        self.outputs = 0.  # collect y_pred batches
        self.path = path
        
        # the shape of these 2 variables will change according to batch shape
        # to handle the "last batch", specify `validate_shape=False`
        self.var_y_true = tf.Variable(0., validate_shape=False)
        self.var_y_pred = tf.Variable(0., validate_shape=False)
        self.var_x_true = tf.Variable(0., validate_shape=False)

    def on_epoch_end(self, epoch, logs=None):
        # evaluate the variables and save them into lists
        self.targets = K.eval(self.var_y_true)
        self.outputs = K.eval(self.var_y_pred)
        self.inputs = K.eval(self.var_x_true)
        plt.imsave(self.path+'/'+str(epoch)+'_targets.png', utils.norm_image(np.squeeze(self.targets[0,])))
        plt.imsave(self.path+'/'+str(epoch)+'_outputs.png', utils.norm_image(np.squeeze(self.outputs[0,])))
        plt.imsave(self.path+'/'+str(epoch)+'_inputs.png', utils.norm_image(np.mean(np.squeeze(self.inputs[0,]),axis=0)))


    