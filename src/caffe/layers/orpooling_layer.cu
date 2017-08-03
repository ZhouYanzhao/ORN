// ORPooling 
#include <vector>

#include "caffe/layers/orpooling_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ORPoolingForward(
    const unsigned int nthreads, 
    const Dtype* bottom_data,
    Dtype* top_data,
    unsigned int* indices_data,
    const unsigned int nOrientation,
    const unsigned int nBatch,
    const unsigned int nChannel,
    const unsigned int nEntry)
{
  CUDA_KERNEL_LOOP(n, nthreads) {
    const unsigned int l = n % nEntry;
    const unsigned int j = (n / nEntry) % nChannel;
    const unsigned int i = n / nEntry / nChannel;
    unsigned int k;
    Dtype max_val = -999999;
    unsigned int* indice = indices_data + n;
    Dtype* target = top_data + n;
    // find maximum
    for (k = 0; k < nOrientation; ++k) {
      const unsigned int src_offset = i * (nChannel * nOrientation * nEntry) 
                                    + j * (nOrientation * nEntry) 
                                    + k * nEntry 
                                    + l;
      const Dtype val = *(bottom_data + src_offset);
      if (val > max_val) {
        max_val = val;
        *indice = src_offset;
      }
    }
    *target = max_val;
  }
}

template <typename Dtype>
void ORPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  unsigned int* indices_data = indices.mutable_gpu_data();
  const unsigned int nBatch = top[0]->shape(0);
  const unsigned int nChannel = top[0]->shape(1);
  const unsigned nEntry = top[0]->shape(2) * top[0]->shape(3);
  const unsigned int count = nBatch * nChannel * nEntry;
  // NOLINT_NEXT_LINE(whitespace/operators)
  ORPoolingForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, indices_data, nOrientation, nBatch, nChannel, nEntry);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ORPoolingBackward(
    const unsigned int nthreads, 
    Dtype* bottom_diff,
    const Dtype* top_diff,
    const unsigned int* indices_data)
{
  CUDA_KERNEL_LOOP(n, nthreads) {
    Dtype* target = bottom_diff + *(indices_data + n);
    *target = *(top_diff + n);
  }
}

template <typename Dtype>
void ORPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const unsigned int* indices_data = indices.gpu_data();
    const unsigned int count = top[0]->shape(0) * top[0]->shape(1) * 
                               top[0]->shape(2) * top[0]->shape(3);
    caffe_gpu_memset(sizeof(Dtype) * (count * nOrientation), 0, bottom_diff);
    // NOLINT_NEXT_LINE(whitespace/operators)
    ORPoolingBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_diff, top_diff, indices_data);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ORPoolingLayer);

}  // namespace caffe