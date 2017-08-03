#ifndef CAFFE_BASE_CONVOLUTION_LAYER_HPP_
#define CAFFE_BASE_CONVOLUTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"

namespace caffe {

/**
 * @brief Abstract base class that factors out the BLAS code common to
 *        ConvolutionLayer and DeconvolutionLayer.
 */
template <typename Dtype>
class BaseConvolutionLayer : public Layer<Dtype> {
 public:
  explicit BaseConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

 protected:
  // Helper functions that abstract away the column buffer and gemm arguments.
  // The last argument in forward_cpu_gemm is so that we can skip the im2col if
  // we just called weight_cpu_gemm with the same input.
  void forward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  void forward_cpu_bias(Dtype* output, const Dtype* bias);
  void backward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output);
  void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype*
      weights);
  void backward_cpu_bias(Dtype* bias, const Dtype* input);

#ifndef CPU_ONLY
  void forward_gpu_gemm(const Dtype* col_input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  void forward_gpu_bias(Dtype* output, const Dtype* bias);
  void backward_gpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* col_output);
  void weight_gpu_gemm(const Dtype* col_input, const Dtype* output, Dtype*
      weights);
  void backward_gpu_bias(Dtype* bias, const Dtype* input);
#endif

  /// @brief The spatial dimensions of the input.
  inline int input_shape(int i) {
    return (*bottom_shape_)[channel_axis_ + i];
  }
  // reverse_dimensions should return true iff we are implementing deconv, so
  // that conv helpers know which dimensions are which.
  virtual bool reverse_dimensions() = 0;
  // Compute height_out_ and width_out_ from other parameters.
  virtual void compute_output_shape() = 0;

  /// @brief The spatial dimensions of a filter kernel.
  Blob<int> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  Blob<int> stride_;
  /// @brief The spatial dimensions of the padding.
  Blob<int> pad_;
  /// @brief The spatial dimensions of the dilation.
  Blob<int> dilation_;
  /// @brief The spatial dimensions of the convolution input.
  Blob<int> conv_input_shape_;
  /// @brief The spatial dimensions of the col_buffer.
  vector<int> col_buffer_shape_;
  /// @brief The spatial dimensions of the output.
  vector<int> output_shape_;
  const vector<int>* bottom_shape_;

  int num_spatial_axes_;
  int bottom_dim_;
  int top_dim_;

  int channel_axis_;
  int num_;
  int channels_;
  int group_;
  int out_spatial_dim_;
  int weight_offset_;
  int num_output_;
  bool bias_term_;
  bool is_1x1_;
  bool force_nd_im2col_;

  // Active rotating filters parameters
  int kernelSize;
  // @brief The number of Oriented feature maps (output)
  int nOutputPlane;
  // @brief The number of Oriented feature maps (input)
  int nInputPlane;
  // @brief The number of ARF orientation channels.
  int nOrientation;
  // @brief The indices for ARF rotation mapping.
  Blob<unsigned int> indices;

 private:
  // generate indices for ARF rotation mapping
  inline void init_indices() {
    int layer_4[] = {
      1,2,3,4,
      2,3,4,1,
      3,4,1,2,
      4,1,2,3
    };
    int kernel_4x1[] = {
      1,
      1,
      1,
      1
    };
    int kernel_4x3[] = {
      1,2,3,4,5,6,7,8,9,
      3,6,9,2,5,8,1,4,7,
      9,8,7,6,5,4,3,2,1,      
      7,4,1,8,5,2,9,6,3
    };
    int layer_8[] = {
      1,2,3,4,5,6,7,8,
      2,3,4,5,6,7,8,1,
      3,4,5,6,7,8,1,2,
      4,5,6,7,8,1,2,3,
      5,6,7,8,1,2,3,4,
      6,7,8,1,2,3,4,5,
      7,8,1,2,3,4,5,6,
      8,1,2,3,4,5,6,7,
    };
    int kernel_8x1[] = {
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1
    };
    int kernel_8x3[] = {
      1,2,3,4,5,6,7,8,9,
      2,3,6,1,5,9,4,7,8,
      3,6,9,2,5,8,1,4,7,
      6,9,8,3,5,7,2,1,4,
      9,8,7,6,5,4,3,2,1,
      8,7,4,9,5,1,6,3,2,
      7,4,1,8,5,2,9,6,3,
      4,1,2,7,5,3,8,9,6
    };
    int* layer_indices;
    int* kernel_indices;
    int col_count;
    if (nOrientation == 4) {
      layer_indices = layer_4;
      if (kernelSize == 1) {
        kernel_indices = kernel_4x1;
        col_count = 1;
      }
      else {
        kernel_indices = kernel_4x3;
        col_count = 9;
      }
    }
    else {
      layer_indices = layer_8;
      if (kernelSize == 1) {
        kernel_indices = kernel_8x1;
        col_count = 1;
      }
      else {
        kernel_indices = kernel_8x3;
        col_count = 9;
      }
    }
    const int shape[] = {nOrientation, kernelSize, kernelSize, nOrientation};
    vector<int> indices_shape(shape, shape + sizeof(shape) / sizeof(int));
    indices.Reshape(indices_shape);
    unsigned int* indices_data = indices.mutable_cpu_data();
    for (int i = 0; i < nOrientation; ++i) {
      for (int j = 0; j < kernelSize * kernelSize; ++j) {
        for (int k = 0; k < nOrientation; ++k) {
          int layer = layer_indices[k * nOrientation + i];
          int kernel = kernel_indices[k * col_count + j];
          int idx = ((i * (kernelSize * kernelSize) + j + 1) - 1) * nOrientation + k;
          indices_data[idx] = (layer - 1) * kernelSize * kernelSize + kernel - 1;
        }
      }
    }
  }
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_cpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), col_buff);
    }
  }
  inline void conv_col2im_cpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_cpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    } else {
      col2im_nd_cpu(col_buff, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), data);
    }
  }
#ifndef CPU_ONLY
  inline void conv_im2col_gpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_gpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_gpu(data, num_spatial_axes_, num_kernels_im2col_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(),
          stride_.gpu_data(), dilation_.gpu_data(), col_buff);
    }
  }
  inline void conv_col2im_gpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_gpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    } else {
      col2im_nd_gpu(col_buff, num_spatial_axes_, num_kernels_col2im_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(),
          dilation_.gpu_data(), data);
    }
  }
#endif

  int num_kernels_im2col_;
  int num_kernels_col2im_;
  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
  int kernel_dim_;
  int col_offset_;
  int output_offset_;

  Blob<Dtype> col_buffer_;
  Blob<Dtype> bias_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_BASE_CONVOLUTION_LAYER_HPP_
