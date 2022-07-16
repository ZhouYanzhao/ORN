  
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>


// Kernels
template <typename scalar_t>
__global__ void mapping_rotate_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> weight_flatten,
    const torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits> indices_flatten,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output,
    const uint16_t nInputPlane,
    const uint16_t nEntry,
    const uint16_t nRotation,
    const uint32_t count
  ) {
  // index
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < count) {
    const uint16_t l = n % nEntry;
    const uint16_t j = (n / nEntry) % nInputPlane;
    const uint16_t i = n / nEntry / nInputPlane;
    const scalar_t val = weight_flatten[n];
    for (uint16_t k = 0; k < nRotation; k++) {
      const uint16_t index = (uint16_t)indices_flatten[l][k] - 1;
      output[i][k][j][index] = val;
    }
  }
}

template <typename scalar_t>
__global__ void mapping_align_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> gradOutput,
    const torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits> indices_flatten,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> gradWeight_flatten,
    const uint16_t nInputPlane,
    const uint16_t nEntry,
    const uint16_t nRotation,
    const uint32_t count
  ) {
  // index
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < count) {
    const uint16_t l = n % nEntry;
    const uint16_t j = (n / nEntry) % nInputPlane;
    const uint16_t i = n / nEntry / nInputPlane;
    scalar_t val = 0.0;
    for (uint16_t k = 0; k < nRotation; k++) {
      const uint16_t index = (uint16_t)indices_flatten[l][k] - 1;
      val += gradOutput[i][k][j][index];
    }
    gradWeight_flatten[n] = val;
  }
}

// Host functions
torch::Tensor mapping_rotate_generic_cuda(torch::Tensor weight, torch::Tensor indices) {
  const auto nOutputPlane = weight.size(0);
  const auto nInputPlane = weight.size(1);
  const auto nOrientation = weight.size(2);
  const auto kH = weight.size(3);
  const auto kW = weight.size(4);
  const auto nRotation = indices.size(3);
  const auto nEntry = nOrientation * kH * kW;
  const uint32_t count = nOutputPlane * nInputPlane * nEntry;
  auto output = torch::empty({nOutputPlane, nRotation, nInputPlane, nEntry}, weight.options());
  auto weight_flatten = weight.view({-1});
  auto indices_flatten = indices.view({nOrientation * kH * kW, nRotation});
  
  const int batch_size = 1;
  const int threads = 1024;
  const dim3 blocks((count + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(weight.type(), "mapping_rotate_cuda", ([&] {
    mapping_rotate_kernel<scalar_t><<<blocks, threads>>>(
        weight_flatten.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
        indices_flatten.packed_accessor32<uint8_t, 2, torch::RestrictPtrTraits>(),
        output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        nInputPlane,
        nEntry,
        nRotation,
        count);
  }));

  return output.view({nOutputPlane * nRotation, nInputPlane * nOrientation, kH, kW});
}

torch::Tensor mapping_align_generic_cuda(torch::Tensor gradOutput, torch::Tensor indices) {
  const auto nOrientation = indices.size(0);
  const auto kH = indices.size(1);
  const auto kW = indices.size(2);
  const auto nRotation = indices.size(3);
  const auto nOutputPlane = gradOutput.size(0) / nRotation;
  const auto nInputPlane = gradOutput.size(1) / nOrientation;
  const auto nEntry = nOrientation * kH * kW;
  const uint32_t count = nOutputPlane * nInputPlane * nEntry;
  gradOutput = gradOutput.view({nOutputPlane, nRotation, nInputPlane, nEntry});
  auto gradWeight = torch::empty({nOutputPlane, nInputPlane, nOrientation, kH, kW}, gradOutput.options());
  auto gradWeight_flatten = gradWeight.view({-1});
  auto indices_flatten = indices.view({nOrientation * kH * kW, nRotation});
  
  const int batch_size = 1;
  const int threads = 1024;
  const dim3 blocks((count + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(gradOutput.type(), "mapping_align_cuda", ([&] {
    mapping_align_kernel<scalar_t><<<blocks, threads>>>(
        gradOutput.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        indices_flatten.packed_accessor32<uint8_t, 2, torch::RestrictPtrTraits>(),
        gradWeight_flatten.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
        nInputPlane,
        nEntry,
        nRotation,
        count);
  }));

  return gradWeight;
}
