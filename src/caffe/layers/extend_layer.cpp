// Sin neuron activation function layer.
// Adapted from TanH layer which was adapted from the ReLU layer code written by Yangqing Jia

#include <vector>

#include "caffe/layers/extend_layer.hpp"

namespace caffe {

template <typename Dtype>
void ExtendLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  nOrientation = this->layer_param_.orn_param().orientations();
  CHECK_GT(nOrientation, 1) << "orientations should be greater than 1";
}

template <typename Dtype>
void ExtendLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num_axes(), 4)
    << "shape of buttom must be nxcxhxw, you are using "
    << bottom[0]->shape_string();
  vector<int> top_shape = bottom[0]->shape();
  // extend channels
  top_shape[1] *= nOrientation;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void ExtendLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) { 
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
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
void ExtendLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) { 
  // Not implemented
}
  
#ifdef CPU_ONLY
STUB_GPU(ExtendLayer);
#endif

INSTANTIATE_CLASS(ExtendLayer);
REGISTER_LAYER_CLASS(Extend);

}  // namespace caffe    