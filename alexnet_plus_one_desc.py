#!/usr/bin/env python

import numpy as np
import os
from skimage import io
from skimage import transform
import caffe

def create_caffenet(model_def_file, pretrained_model):
    caffenet = caffe.Net(model_def_file, pretrained_model)
    caffenet.set_phase_test()
    caffenet.set_mode_cpu()
    return caffenet

MODEL_FILE = '/media/big_disk/installers_old/caffe_lowMemConv/examples/imagenet/imagenet_deploy.prototxt'
PRETRAINED = 'media/big_disk/installers_old/caffe_lowMemConv/alexnet_train_iter_470000'

net = create_caffenet(MODEL_FILE, PRETRAINED)
IMAGE_FILE = '/media/big_disk/installers_old/caffe/voc-release5/cachedir/VOC2007/JPEGImages/000023.jpg'

predictions = net.predict(IMAGE_FILE)
print predictions
 

