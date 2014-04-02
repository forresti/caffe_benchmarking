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

def prepare_image(img):
    if img.ndim == 2:
        img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    img = (img*255)[:, :, ::-1] #RGB -> BGR
    #img -= IMAGENET_MEAN #don't do data centering for now.
    img = img.transpose((2,0,1))
    img = np.ascontiguousarray(img)
    img = np.expand_dims(img, axis=0) #(3,227,227) -> (1,3,227,227)
    img = img.astype(np.float32)
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

img = io.imread(IMAGE_FILE)
input_blob = img[0:227, 0:227, :]
input_blob = prepare_image(input_blob)
output_blob_dims = get_last_blob_dims(caffenet)
output_blobs = [np.empty((output_blob_dims), dtype=np.float32)] 
caffenet.Forward([input_blob], output_blobs) 

caffenet_227x227_data = caffenet.blobs
#layerNames = caffenet.blobs.keys()
#conv2_data = caffenet.blobs['conv2']
#print conv2_data.data
#SUCCESS. was able to pull out an individual layer's data.


#embed()

