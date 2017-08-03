// Sin neuron activation function layer.
// Adapted from TanH layer which was adapted from the ReLU layer code written by Yangqing Jia

#include <vector>

#include "caffe/layers/sin_layer.hpp"

namespace caffe {
  template <typename Dtype>
  void SinLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) 
  { 
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    printf("top: %d, bottom: %d\n", top[0]->count(), bottom[0]->count());
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      top_data[i] = sin(bottom_data[i]);
    }
  }

  template <typename Dtype>
  void SinLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<Dtype>*>& bottom) 
  { 
    if (propagate_down[0]) {
      const Dtype* bottom_data = bottom[0]->cpu_data();
      const Dtype* top_diff = top[0]->cpu_diff();
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      const int count = bottom[0]->count();
      Dtype bottom_datum;
      for (int i = 0; i < count; ++i) {
        bottom_datum = bottom_data[i];
        bottom_diff[i] = top_diff[i] * cos(bottom_datum);
      }
    }
  }
  
#ifdef CPU_ONLY
STUB_GPU(SinLayer);
#endif

INSTANTIATE_CLASS(SinLayer);
REGISTER_LAYER_CLASS(Sin);

}  // namespace caffe    