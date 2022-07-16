  
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>


#define INF 2.0e+10F

// Kernels
template <typename scalar_t>
__global__ void align_feature_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits> mainDirection,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> output,
    const uint16_t nFeature,
    const uint8_t nOrientation,
    const uint32_t count
  ) {
  // index
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < count) {
    const uint16_t j = n % nFeature;
    const uint16_t i = n / nFeature;

    // Find main directions
    uint8_t l;
    scalar_t maxVal = -INF;
    for (l = 0; l < nOrientation; l++) {
      const scalar_t val = input[i][j][l];
      if (val > maxVal) {
          maxVal = val;
          mainDirection[i][j] = l;
      }
    }
    // Align features
    for (l = 0; l < nOrientation; l++) {
      const uint8_t alignedIndex = (l - mainDirection[i][j] + nOrientation) % nOrientation;
      output[i][j][alignedIndex] = input[i][j][l];
    }
  }
}

template <typename scalar_t>
__global__ void unalign_feature_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> gradOutput,
    const torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits> mainDirection,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> gradInput,
    const uint16_t nFeature,
    const uint8_t nOrientation,
    const uint32_t count
  ) {
  // index
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < count) {
    const uint16_t j = n % nFeature;
    const uint16_t i = n / nFeature;

    // Align features
    for (auto l = 0; l < nOrientation; l++) {
      const uint8_t alignedIndex = (l + mainDirection[i][j]) % nOrientation;
      gradInput[i][j][alignedIndex] = gradOutput[i][j][l];
    }
  }
}

// Host functions
std::vector<torch::Tensor> align_feature_generic_cuda(torch::Tensor input, uint8_t nOrientation) {
  const auto nBatch = input.size(0);
  const auto nChannel = input.size(1);
  const auto nFeature = nChannel / nOrientation;
  const auto kH = input.size(2);
  const auto kW = input.size(3);
  AT_ASSERTM((kH == 1) && (kW == 1), "mH x mW should be 1x1.");

  input = input.view({nBatch, nFeature, nOrientation});
  auto mainDirection = torch::empty({nBatch, nFeature}, input.options().dtype(torch::kUInt8));
  auto output = torch::empty({nBatch, nFeature, nOrientation}, input.options());

  const uint32_t count = nBatch * nFeature;
  const int batch_size = 1;
  const int threads = 1024;
  const dim3 blocks((count + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "align_feature_cuda", ([&] {
    align_feature_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        mainDirection.packed_accessor32<uint8_t, 2, torch::RestrictPtrTraits>(),
        output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        nFeature,
        nOrientation,
        count);
  }));

  return {output.view({nBatch, nChannel, 1, 1}), mainDirection};
}

torch::Tensor unalign_feature_generic_cuda(torch::Tensor gradOutput, torch::Tensor mainDirection, uint8_t nOrientation) {
  const auto nBatch = gradOutput.size(0);
  const auto nChannel = gradOutput.size(1);
  const auto nFeature = nChannel / nOrientation;
  const auto kH = gradOutput.size(2);
  const auto kW = gradOutput.size(3);
  AT_ASSERTM((kH == 1) && (kW == 1), "kH x kW should be 1x1.");

  gradOutput = gradOutput.view({nBatch, nFeature, nOrientation});
  auto gradInput = torch::empty({nBatch, nFeature, nOrientation}, gradOutput.options());

  const uint32_t count = nBatch * nFeature;
  const int batch_size = 1;
  const int threads = 1024;
  const dim3 blocks((count + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(gradOutput.type(), "unalign_feature_cuda", ([&] {
    unalign_feature_kernel<scalar_t><<<blocks, threads>>>(
        gradOutput.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        mainDirection.packed_accessor32<uint8_t, 2, torch::RestrictPtrTraits>(),
        gradInput.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        nFeature,
        nOrientation,
        count);
  }));

  return gradInput.view({nBatch, nChannel, 1, 1});
}
