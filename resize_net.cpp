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

int main(int argc, char** argv) {
  cudaSetDevice(0);
  Caffe::set_mode(Caffe::CPU);
  Caffe::set_phase(Caffe::TEST);
  int repeat = 5;

  NetParameter net_param;
  ReadProtoFromTextFile(argv[1],
      &net_param);
  Net<float> caffe_net(net_param);

  LOG(ERROR) << "Performing Forward";
  caffe_net.Forward(vector<Blob<float>*>());

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();


  for (int i = 0; i < layers.size(); ++i) {
    const string& layername = layers[i]->layer_param().name();
    layers[i]->Forward(bottom_vecs[i], &top_vecs[i]);

    //TODO: print layer dims.
  }

  return 0;
}
