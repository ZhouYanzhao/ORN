// Sin neuron activation function layer.
// Adapted from TanH layer which was adapted from the ReLU layer code written by Yangqing Jia

#include <vector>

#include "caffe/layers/sin_layer.hpp"

namespace caffe {
  template <typename Dtype>
  __global__ void SinForward(const int n, const Dtype* in, Dtype* out) {
    CUDA_KERNEL_LOOP(index, n) {
      out[index] = sin(in[index]);
    }
  }

  template <typename Dtype>
  void SinLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SinForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, top_data);
    CUDA_POST_KERNEL_CHECK;
  }

  template <typename Dtype>
  __global__ void SinBackward(const int n, const Dtype* in_diff,
      const Dtype* out_data, Dtype* out_diff) {
    CUDA_KERNEL_LOOP(index, n) {
      Dtype sinx = out_data[index];
      out_diff[index] = in_diff[index] * cos(sinx);
    }
  }

  template <typename Dtype>
  void SinLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[0]) {
      const Dtype* bottom_data = bottom[0]->gpu_data();
      const Dtype* top_diff = top[0]->gpu_diff();
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      const int count = bottom[0]->count();
      // NOLINT_NEXT_LINE(whitespace/operators)
      SinBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, top_diff, bottom_data, bottom_diff);
      CUDA_POST_KERNEL_CHECK;
    }
  }
  INSTANTIATE_LAYER_GPU_FUNCS(SinLayer);

}  // namespace caffe