#!/usr/bin/env python

import numpy as np
import os
from skimage import io
from skimage import transform
import caffe
from IPython import embed

def create_caffenet(model_def_file, pretrained_model):
    caffenet = caffe.Net(model_def_file, pretrained_model)
    caffenet.set_phase_test()
    caffenet.set_mode_cpu()
    return caffenet

def prepare_image(filename):
    img = io.imread(filename)
    if img.ndim == 2:
        img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    img.shape
    img = (img*255)[:, :, ::-1] #RGB -> BGR
    #img -= IMAGENET_MEAN #don't do data centering for now.
    return img

#get the dims of the output blob for the final layer, after initializing caffe.
def get_last_blob_dims(caffenet):
    layerName = caffenet.blobs.keys()[-1]
    return np.shape(caffenet.blobs[layerName].data)


#MODEL_FILE = '/media/big_disk/installers_old/caffe_lowMemConv/examples/imagenet/imagenet_deploy.prototxt'
MODEL_FILE = '/media/big_disk/installers_old/caffe/examples/imagenet_deploy_batchsize1_output_conv5.prototxt'
PRETRAINED = '/media/big_disk/installers_old/caffe_lowMemConv/examples/alexnet_train_iter_470000'
IMAGE_FILE = '/media/big_disk/installers_old/caffe/voc-release5/cachedir/VOC2007/JPEGImages/000023.jpg'

caffenet = create_caffenet(MODEL_FILE, PRETRAINED)

input_blob = prepare_image(IMAGE_FILE)
input_blob = input_blob.transpose((2,0,1))
input_blob = input_blob[:, 0:227, 0:227] #crop
input_blob = np.ascontiguousarray(input_blob)
input_blob = np.expand_dims(input_blob, axis=0) #(3,227,227) -> (1,3,227,227)
input_blob = input_blob.astype(np.float32)
output_blob_dims = get_last_blob_dims(caffenet)
output_blobs = [np.empty((output_blob_dims), dtype=np.float32)] 
caffenet.Forward([input_blob], output_blobs) 

layerNames = caffenet.blobs.keys()
conv2_data = caffenet.blobs['conv2']
#print conv2_data.data
#SUCCESS. was able to pull out an individual layer's data.

#TODO: run with 227x227 input and 243x243 input, and save all layers. compare each layer.

#embed()

