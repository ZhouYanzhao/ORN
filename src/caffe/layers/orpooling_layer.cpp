// Sin neuron activation function layer.
// Adapted from TanH layer which was adapted from the ReLU layer code written by Yangqing Jia

#include <vector>

#include "caffe/layers/orpooling_layer.hpp"

namespace caffe {

template <typename Dtype>
void ORPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  nOrientation = this->layer_param_.orn_param().orientations();
  CHECK_GT(nOrientation, 1) << "orientations should be greater than 1";
}

template <typename Dtype>
void ORPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num_axes(), 4)
    << "shape of buttom must be nxcxhxw, you are using "
    << bottom[0]->shape_string();
  vector<int> top_shape = bottom[0]->shape();
  CHECK_EQ(top_shape[1] % nOrientation, 0) 
    << "the number of input channels must be a multiple of nOrientation";
  // pooling across orientations
  top_shape[1] /= nOrientation;
  top[0]->Reshape(top_shape);
  indices.Reshape(top_shape);
}

template <typename Dtype>
void ORPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  unsigned int* indices_data = indices.mutable_cpu_data();
  const unsigned int n = top[0]->shape(0);
  const unsigned int c = top[0]->shape(1);
  const unsigned count = top[0]->shape(2) * top[0]->shape(3);
  Dtype max_val;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; ++j) {
      const unsigned int factor1 = i * c * count + j * count;
      for (int l = 0; l < count; ++l) {
        max_val = -999999;
        const unsigned int dst_offset = factor1 + l;
        unsigned int* indice = indices_data + dst_offset;
        Dtype* target = top_data + dst_offset;
        const unsigned int factor2 = i * c * nOrientation * count 
                                   + j * nOrientation * count
                                   + l;
        // get the max value
        for (int k = 0; k < nOrientation; ++k) {
          const unsigned int src_offset = factor2 + k * count;
          const Dtype val = *(bottom_data + src_offset);
          if (val > max_val) {
            max_val = val;
            *indice = src_offset;
          }
        }
        *target = max_val;
      }
    }
  }
}

template <typename Dtype>
void ORPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) { 
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const unsigned int* indices_data = indices.cpu_data();
    const unsigned count = top[0]->shape(0) * top[0]->shape(1) *
                           top[0]->shape(2) * top[0]->shape(3);
    caffe_memset(sizeof(Dtype) * (nOrientation * count), 0, bottom_diff);
    for (int n = 0; n < count; ++n) {
      Dtype* target = bottom_diff + *(indices_data + n);
      *target = *(top_diff + n);
    }
  }
}
  
#ifdef CPU_ONLY
STUB_GPU(ORPoolingLayer);
#endif

INSTANTIATE_CLASS(ORPoolingLayer);
REGISTER_LAYER_CLASS(ORPooling);

}  // namespace caffe    