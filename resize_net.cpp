// Copyright 2013 Yangqing Jia

#include <cuda_runtime.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>

#include <cstring>
#include <ctime>
#include <cstdio>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/solver.hpp"
#include <sys/time.h>

using namespace caffe;

double read_timer(){
    struct timeval start;
    gettimeofday( &start, NULL );
    return (double)((start.tv_sec) + 1.0e-6 * (start.tv_usec)) * 1000; //in milliseconds
}

void print_layer_sizes(Net<float> &caffe_net){
    caffe_net.Forward(vector<Blob<float>*>());
    const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
    vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
    vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();

     for (int i = 0; i < layers.size(); ++i) {
        //TODO: skip data layer (I think the data layer doesn't have a valid 'bottom' vector)

        const string& layername = layers[i]->layer_param().name();
        //layers[i]->Forward(bottom_vecs[i], &top_vecs[i]); //can run the layers, but not necessary here.

        LOG(ERROR) << "layer name: " << layername;

        LOG(ERROR) << "    bottom (input): num=" << bottom_vecs[i][0]->num() << 
                      " channels=" << bottom_vecs[i][0]->channels() << 
                      " height=" << bottom_vecs[i][0]->height() <<
                      " width=" << bottom_vecs[i][0]->width(); 


        LOG(ERROR) << "    top (output): num=" << top_vecs[i][0]->num() << 
                      " channels=" << top_vecs[i][0]->channels() << 
                      " height=" << top_vecs[i][0]->height() <<
                      " width=" << top_vecs[i][0]->width(); 
    }
}

int main(int argc, char** argv) {
    cudaSetDevice(0);
    Caffe::set_mode(Caffe::CPU);
    Caffe::set_phase(Caffe::TEST);

    NetParameter net_param;
    ReadProtoFromTextFile(argv[1],
            &net_param);
    #if 0
    Net<float> caffe_net(net_param); //initialize net (and print most of the blob sizes too)
    //print_layer_sizes(caffe_net); //default size from prototxt file.

    //0: num, 1: channels, 2: height, 3: width
    net_param.set_input_dim(2, 227); //height
    net_param.set_input_dim(3, 227); //width
    Net<float> caffe_net_resized(net_param); //initialize net (and print most of the blob sizes too)
    //print_layer_sizes(caffe_net_resized); //updated sizes
    #endif

    // inputSz=214 -> conv5=12x12x256

    // inputSz=215 -> conv5=13x13x256
    // inputSz=230 -> conv5=13x13x256

    // inputSz=231 -> conv5=14x14x256
    // inputSz=246 -> conv5=14x14x256

    // inputSz=247 -> conv5=15x15x256

    //look for size where we go from conv5=13x13x256 to conv5=14x14x256.
    //for(int inputSz = 210; inputSz<250; inputSz++)
    for(int inputSz = 350; inputSz<400; inputSz++)
    {
        net_param.set_input_dim(2, inputSz); //height
        net_param.set_input_dim(3, inputSz); //width
        Net<float> caffe_net_resized(net_param); //initialize net (and print most of the blob sizes too)
        print_layer_sizes(caffe_net_resized); //includes bottom and top printouts
    }

    return 0;
}
