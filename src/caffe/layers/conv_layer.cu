#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void RotateKernel(
    const unsigned int nthreads, 
    const unsigned int* indices_data,
    const unsigned int nInputPlane,
    const unsigned int nOutputPlane,
    const unsigned int nOrientation,
    const unsigned int nEntry,
    Dtype* weight_data) 
{
  CUDA_KERNEL_LOOP(n, nthreads) {
    const unsigned int l = n % nEntry;
    const unsigned int j = (n / nEntry) % nInputPlane;
    const unsigned int i = n / nEntry / nInputPlane;
    unsigned int k;
    const Dtype val = *(weight_data + i * (nOrientation * nInputPlane * nEntry)
                                    + j * (nEntry)
                                    + l);
    for (k = 1; k < nOrientation; k++) {
      const unsigned int index = (unsigned int)(*(indices_data + l * nOrientation + k));
      Dtype *target = weight_data + i * (nOrientation * nInputPlane * nEntry)
                                  + k * (nInputPlane * nEntry)
                                  + j * (nEntry)
                                  + index;
      *target = val;
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::RotateARF_gpu() {
  const int kH = this->kernelSize;
  const int nOrientation = this->nOrientation;
  const int nOutputPlane = this->nOutputPlane;
  const int nInputPlane = this->nInputPlane;
  const unsigned int* indices_data = this->indices.gpu_data();
  Dtype* weight_data = this->blobs_[0]->mutable_gpu_data();
  const unsigned int nEntry = nOrientation * kH * kH;
  const unsigned int count = nOutputPlane * nInputPlane * nEntry;
  // NOLINT_NEXT_LINE(whitespace/operators)
  RotateKernel<Dtype> <<< CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >>>
      (count, indices_data, nInputPlane, nOutputPlane, nOrientation, nEntry, weight_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void AlignKernel(
    const unsigned int nthreads, 
    const unsigned int* indices_data,
    const unsigned int nInputPlane,
    const unsigned int nOutputPlane,
    const unsigned int nOrientation,
    const unsigned int nEntry,
    Dtype* weight_diff_data) 
{
  CUDA_KERNEL_LOOP(n, nthreads) {
    const unsigned int l = n % nEntry;
    const unsigned int j = (n / nEntry) % nInputPlane;
    const unsigned int i = n / nEntry / nInputPlane;
    unsigned int k;
    Dtype* val = weight_diff_data + i * (nOrientation * nInputPlane * nEntry)
                                  + j * (nEntry)
                                  + l;
    for (k = 1; k < nOrientation; k++) {
      const unsigned int index = (unsigned int)(*(indices_data + l * nOrientation + k));
      Dtype *target = weight_diff_data + i * (nOrientation * nInputPlane * nEntry)
                                       + k * (nInputPlane * nEntry)
                                       + j * (nEntry)
                                       + index;
      *val = *val + *target;
      *target = 0;
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::AlignARF_gpu() {
  const int kH = this->kernelSize;
  const int nOrientation = this->nOrientation;
  const int nOutputPlane = this->nOutputPlane;
  const int nInputPlane = this->nInputPlane;
  const unsigned int* indices_data = this->indices.gpu_data();
  Dtype* weight_diff_data = this->blobs_[0]->mutable_gpu_diff();
  const unsigned int nEntry = nOrientation * kH * kH;
  const unsigned int count = nOutputPlane * nInputPlane * nEntry;
  // NOLINT_NEXT_LINE(whitespace/operators)
  AlignKernel<Dtype> <<< CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >>>
      (count, indices_data, nInputPlane, nOutputPlane, nOrientation, nEntry, weight_diff_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  if (this->nOrientation > 1) {
    // generate rotated ARFs
    this->RotateARF_gpu();
  }
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
          // Align ARFs
          if (this->nOrientation > 1) {
            this->AlignARF_gpu();
          }
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);
template __global__ void RotateKernel<float>(
    const unsigned int nthreads, 
    const unsigned int* indices_data,
    const unsigned int nInputPlane,
    const unsigned int nOutputPlane,
    const unsigned int nOrientation,
    const unsigned int nEntry,
    float* weight_data);
template __global__ void RotateKernel<double>(
    const unsigned int nthreads, 
    const unsigned int* indices_data,
    const unsigned int nInputPlane,
    const unsigned int nOutputPlane,
    const unsigned int nOrientation,
    const unsigned int nEntry,
    double* weight_data);
template void ConvolutionLayer<float>::RotateARF_gpu(); 
template void ConvolutionLayer<double>::RotateARF_gpu();
template __global__ void AlignKernel<float>(
    const unsigned int nthreads, 
    const unsigned int* indices_data,
    const unsigned int nInputPlane,
    const unsigned int nOutputPlane,
    const unsigned int nOrientation,
    const unsigned int nEntry,
    float* weight_data);
template __global__ void AlignKernel<double>(
    const unsigned int nthreads, 
    const unsigned int* indices_data,
    const unsigned int nInputPlane,
    const unsigned int nOutputPlane,
    const unsigned int nOrientation,
    const unsigned int nEntry,
    double* weight_data);
template void ConvolutionLayer<float>::AlignARF_gpu(); 
template void ConvolutionLayer<double>::AlignARF_gpu();
}  // namespace caffe
