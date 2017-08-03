// Sin neuron activation function layer.
// Adapted from TanH layer which was adapted from the ReLU layer code written by Yangqing Jia

#include <vector>

#include "caffe/layers/extend_layer.hpp"

namespace caffe {

template <typename Dtype>
void ExtendLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const unsigned int n = bottom[0]->shape(0);
  const unsigned int c = bottom[0]->shape(1);
  const unsigned count = bottom[0]->shape(2) * bottom[0]->shape(3);
  // for optimization
  const unsigned int factor1 = c * count;
  const unsigned int factor2 = c * nOrientation * count;
  const unsigned int factor3 = nOrientation * count;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; ++j) {
      const unsigned int src_offset = i * factor1 + j * count;
      for (int k = 0; k < nOrientation; ++k) {
        caffe_copy(count, 
          bottom_data + src_offset, 
          top_data + i * factor2 + j * factor3 + k * count);
      }
    }
  }
}

template <typename Dtype>
void ExtendLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // Not implemented
}

INSTANTIATE_LAYER_GPU_FUNCS(ExtendLayer);

}  // namespace caffe