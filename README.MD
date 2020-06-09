# Learn2SOFI
This work is based on [TF_Unet](https://github.com/jakeret/tf_unet/tree/master/tf_unet) and [ScaDec](https://github.com/sunyumark/ScaDec-deep-learning-diffractive-tomography).

## Abstract
empty

## Experimental Results
empty

## Usage

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

# start spyder 
spyder &
```


### Download Images / Create Dataset

First you can start downloading random images from biological cells. This can for example be done by using the API from OMERO image browser. Please review individual licneses! 

By using ```01_downloadimages.py```available [here](./SCRIPTS/01_downloadimages.py) donloads images into a folder e.g. ```DATA``` (please specify!). 

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

