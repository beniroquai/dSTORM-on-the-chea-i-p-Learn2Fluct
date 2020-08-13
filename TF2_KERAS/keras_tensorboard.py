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


    

# make the 1 channel input image or disparity map look good within this color map. This function is not necessary for this Tensorboard problem shown as above. Just a function used in my own research project.
def colormap_jet(img):
    return cv2.cvtColor(cv2.applyColorMap(np.uint8(img), 2), cv2.COLOR_BGR2RGB)

class customModelCheckpoint(Callback):
    def __init__(self, log_dir='./logs/tmp/', feed_inputs_display=None):
          super(customModelCheckpoint, self).__init__()
          self.seen = 0
          self.feed_inputs_display = feed_inputs_display
          self.writer = tf.summary.SummaryWriter(log_dir) 

    # this function will return the feeding data for TensorBoard visualization;
    # arguments:
    #  * feed_input_display : [(input_yourModelNeed, left_image, disparity_gt ), ..., (input_yourModelNeed, left_image, disparity_gt), ...], i.e., the list of tuples of Numpy Arrays what your model needs as input and what you want to display using TensorBoard. Note: you have to feed the input to the model with feed_dict, if you want to get and display the output of your model. 
    def custom_set_feed_input_to_display(self, feed_inputs_display):
          self.feed_inputs_display = feed_inputs_display

    # copied from the above answers;
    def make_image(self, numpy_img):
          from PIL import Image
          height, width, channel = numpy_img.shape
          image = Image.fromarray(numpy_img)
          import io
          output = io.BytesIO()
          image.save(output, format='PNG')
          image_string = output.getvalue()
          output.close()
          return tf.Summary.Image(height=height, width=width, colorspace= channel, encoded_image_string=image_string)


    # A callback has access to its associated model through the class property self.model.
    def on_batch_end(self, batch, logs = None):
          logs = logs or {} 
          self.seen += 1
          if self.seen % 200 == 0: # every 200 iterations or batches, plot the costumed images using TensorBorad;
              summary_str = []
              for i in range(len(self.feed_inputs_display)):
                  feature, disp_gt, imgl = self.feed_inputs_display[i]
                  disp_pred = np.squeeze(K.get_session().run(self.model.output, feed_dict = {self.model.input : feature}), axis = 0)
                  #disp_pred = np.squeeze(self.model.predict_on_batch(feature), axis = 0)
                  summary_str.append(tf.Summary.Value(tag= 'plot/img0/{}'.format(i), image= self.make_image( colormap_jet(imgl)))) # function colormap_jet(), defined above;
                  summary_str.append(tf.Summary.Value(tag= 'plot/disp_gt/{}'.format(i), image= self.make_image( colormap_jet(disp_gt))))
                  summary_str.append(tf.Summary.Value(tag= 'plot/disp/{}'.format(i), image= self.make_image( colormap_jet(disp_pred))))

              self.writer.add_summary(tf.Summary(value = summary_str), global_step =self.seen)