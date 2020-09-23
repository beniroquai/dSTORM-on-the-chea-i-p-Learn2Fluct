# Learn2SOFI

With this machine learning algorithm we aim to compute a higher optical resolution based on a temporally fluctuating excitation intensity of fluorophores. The core-idea is derived from methods like super-resolution optical fluctuation imaging - SOFI ([Paper](https://www.pnas.org/content/106/52/22287), [Wiki](https://en.wikipedia.org/wiki/Super-resolution_optical_fluctuation_imaging)). It reconstructs higher spatially frequencies usually lost during the acquisition process.
An alternative is SRRF ([Paper](https://www.nature.com/articles/ncomms12471), [Github](https://github.com/HenriquesLab/NanoJ-SRRF)). Both methods are computational complex and therefore not suitable for low-powered devices such as cellphones or embedded devices.

Here we try to represent the idea of recovering correlated information along time (i.e. fluorescent structures, but no noise) using a convolutional long short term memory network (cLSTM). A digital representation of the microscope forward model produces realistic temporal stacks from arbitrary fluorophore distributions which is the input for the neural network. The reference is the high-resolution output image.

 The python software is partially based on work by [TF_Unet](https://github.com/jakeret/tf_unet/tree/master/tf_unet) and [ScaDec](https://github.com/sunyumark/ScaDec-deep-learning-diffractive-tomography) and works with ```Tensorflow >2.2``` to be used for ```TFLite``` and ```TensorflowJs```.

## Abstract

The preprint is now available with a more in-depth discussion about the network. You can find it on [Biorxiv](https://www.biorxiv.org/content/10.1101/2020.09.04.283085v1).


## Preparation

### Setup the Environment
```
# install the environment
conda create -n tfgpu2 python=3.6
conda activate tfgpu2

# install spyder and other dependencies
conda install -c anaconda spyder
conda install -c anaconda tifffile
pip install matplotlib
pip install git+https://NanoImagingUser:NanoImagingUser@gitlab.com/bionanoimaging/nanoimagingpack
pip install opencv-python
pip install tensorflow==2.3.0
pip install tensorflowjs
pip install json


# start spyder
spyder &
```

### Download Images / Create Dataset

First you can start downloading random images from biological cells. This can for example be done by using the API from OMERO image browser. Please review individual licneses!

By using ```01_downloadimages.py```available [here](./SCRIPTS/01_downloadimages.py) downloads images into a folder e.g. ```DATA``` (please specify!).

### Download Dataset

We provide the training dataset of images used for the publication in a [GDrive folder](https://drive.google.com/drive/folders/1Xv8PxXt7VSZb3fNiXgfviUCi-6OUCqoA?usp=sharing). We do not guarantee to own the rights of the images since they are downloaded from *OMERO* and may have their own license type.


# Introduction

The code in this repository may become a little bit confusing. During the creation of the architecture we had to find the best possible network design to produce good super-resolved results, but keeping it simple and light-weight at the same time. Additionally, the number of different operands is very limited when it comes to deployment on mobile devices using ```TFLite``` or in the browser using ```TensorflowJs```. We created two different network types:


## TFLite to be used on Android cellphones

Tensorflow Lite is a nice way to export trained models to devices such as cellphones or microcontrollers. A brief introduction can be found [here](https://towardsdatascience.com/a-basic-introduction-to-tensorflow-lite-59e480c57292). It is important to keep in mind that [not all operands are supported](https://www.tensorflow.org/lite/guide/ops_compatibility). Especially the RNN used in this framework was tricky to implement. If you have problems to export it, please use a google colab notebook - it seems Tensorflow GPU does not work with the cLSTM layers..

The network implemented in Keras is available in the folder [.\TF2_KERAS](.\TF2_KERAS).

### General Steps

#### Training

Use the file  [keras_train.py](./TF2_KERAS/keras_train.py) and change the variable ```data_dir``` to point to the folder of downloaded ```.TIFF```-images.
After training has succeeded, a model is exported called ```test.hdf5```

Make sure you set the variable ```export_type='tflite'```


#### Exporting

For exporting the file please use the file [keras_exporttflite.py](./TF2_KERAS/keras_exporttflite.py). The resulting file ```XXXX.tflite``` needs to be copied to your android project.


## TensorflowJs to be used in your browser

### General Steps

#### Training

Use the file  [keras_train.py](./TF2_KERAS/keras_train.py) and change the variable ```data_dir``` to point to the folder of downloaded ```.TIFF```-images.
After training has succeeded, a model is exported called ```test.hdf5```

Make sure you set the variable ```export_type='tfjs'```

#### Exporting

For exporting the file please use the file [keras_exportjs.py](./TF2_KERAS/keras_exportjs.py). The resulting file ```XXXX.json``` needs to be copied to your tensorflowjs  project.


## Experimental Results
empty



### Training

The file ```mytrain.py``` creates the network and organises the training. Important parameters are the ```DATA```-folder. Also mark down the name for training results. To review the trianing process you can use ```tensorboard```by typing

```tensorboard --logdir=./``` if you are in the current directory

### Testing

The file ```mytest.py``` loads some sample data and propagates it through the network.

### Convert Graph to TFLite

The file ```myconvert_tflite.py``` is converting a frozen graph in the folder ```XXX/XXX/final/```to a ```*.tflite``` file. This need to be copied to the assets folder in the Android APP (i.e. in Android Studio).

### Setup Android

Please vary the code in the ```AcquireActivity.java``` according to the exported Tensorflow Graph:

```
String mymodelfile = "converted_model128_30.tflite";
String mymodelfile_mean = "converted_model_mean.tflite";
String mymodelfile_stdv = "converted_model_stdv.tflite";

// Tensorflow stuff
int Nx_in = 128;
int Ny_in = Nx_in;
int N_time = 30;
int N_upscale = 2; // Upscalingfactor
```

## Examples

Image Sequence of SOFI experiments:
<p align="center">
<img src="./IMAGES/mytest_obj-2.gif" width="300">
</p>

Reconstructed the result from the network:
<p align="center">
<img src="./IMAGES/mytest_gt-1.png" width="300">
</p>


## Citation
empty
