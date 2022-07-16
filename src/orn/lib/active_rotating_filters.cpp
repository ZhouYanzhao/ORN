#include <vector>
#include <torch/extension.h>


#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

// CUDA declarations
torch::Tensor mapping_rotate_generic_cuda(torch::Tensor weight, torch::Tensor indices);
torch::Tensor mapping_align_generic_cuda(torch::Tensor gradOutput, torch::Tensor indices);

// CPU declarations
template <typename scalar_t>
torch::Tensor mapping_rotate_generic(torch::Tensor weight, torch::Tensor indices) {
  const auto nOutputPlane = weight.size(0);
  const auto nInputPlane = weight.size(1);
  const auto nOrientation = weight.size(2);
  const auto kH = weight.size(3);
  const auto kW = weight.size(4);
  const auto nRotation = indices.size(3);
  const auto nEntry = nOrientation * kH * kW;
  auto output = torch::empty({nOutputPlane, nRotation, nInputPlane, nEntry}, weight.options());
  auto output_accessor = output.accessor<scalar_t, 4>();
  auto weight_flatten = weight.view({-1});
  auto weight_flatten_accessor = weight_flatten.accessor<scalar_t, 1>();
  auto indices_flatten = indices.view({nOrientation * kH * kW, nRotation});
  auto indices_flatten_accessor = indices_flatten.accessor<uint8_t, 2>();
  
  auto offset = 0;
  for (auto i = 0; i < nOutputPlane; i++) {
    for (auto j = 0; j < nInputPlane; j++) {
      for (auto l = 0; l < nEntry; l++) {
        const auto val = weight_flatten_accessor[offset++];
        for (auto k = 0; k < nRotation; k++) {
          const auto index = indices_flatten_accessor[l][k] - 1;
          output_accessor[i][k][j][index] = val;
        }
      }
    }
  }
  return output.view({nOutputPlane * nRotation, nInputPlane * nOrientation, kH, kW});
}

template <typename scalar_t>
torch::Tensor mapping_align_generic(torch::Tensor gradOutput, torch::Tensor indices) {
  const auto nOrientation = indices.size(0);
  const auto kH = indices.size(1);
  const auto kW = indices.size(2);
  const auto nRotation = indices.size(3);
  const auto nOutputPlane = gradOutput.size(0) / nRotation;
  const auto nInputPlane = gradOutput.size(1) / nOrientation;
  const auto nEntry = nOrientation * kH * kW;
  auto gradWeight = torch::empty({nOutputPlane, nInputPlane, nOrientation, kH, kW}, gradOutput.options());
  auto gradWeight_flatten = gradWeight.view({-1});
  auto gradWeight_flatten_accessor = gradWeight_flatten.accessor<scalar_t, 1>();
  gradOutput = gradOutput.view({nOutputPlane, nRotation, nInputPlane, nEntry});
  auto gradOutput_accessor = gradOutput.accessor<scalar_t, 4>();
  auto indices_flatten = indices.view({nOrientation * kH * kW, nRotation});
  auto indices_flatten_accessor = indices_flatten.accessor<uint8_t, 2>();
  
  auto offset = 0;
  for (auto i = 0; i < nOutputPlane; i++) {
    for (auto j = 0; j < nInputPlane; j++) {
      for (auto l = 0; l < nEntry; l++) {
        scalar_t val = 0.0;
        for (auto k = 0; k < nRotation; k++) {
          const auto index = indices_flatten_accessor[l][k] - 1;
          val += gradOutput_accessor[i][k][j][index];
        }
        gradWeight_flatten_accessor[offset++] = val;
      }
    }
  }
  return gradWeight;
}

// C++ interface
torch::Tensor mapping_rotate(torch::Tensor weight, torch::Tensor indices) {
  // Check inputs
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(indices);

  if ((weight.device().is_cuda()) || (indices.device().is_cuda())) {
    // Run on CUDA
    return mapping_rotate_generic_cuda(weight, indices);
  }
  else {
    // Run on CPU
    if (weight.dtype() == torch::kFloat32) {
      return mapping_rotate_generic<float>(weight, indices);
    }
    else if (weight.dtype() == torch::kFloat64) {
      return mapping_rotate_generic<double>(weight, indices);
    }
    else if (weight.dtype() == torch::kFloat16) {
      return mapping_rotate_generic<at::Half>(weight, indices);
    }
    else {
      std::cout << "Not implemented for " << weight.dtype() << std::endl;
    }
  }
}

torch::Tensor mapping_align(torch::Tensor gradOutput, torch::Tensor indices) {
  // Check inputs
  CHECK_CONTIGUOUS(gradOutput);
  CHECK_CONTIGUOUS(indices);

  if ((indices.device().is_cuda()) || (gradOutput.device().is_cuda())) {
    // Run on CUDA
    return mapping_align_generic_cuda(gradOutput, indices);
  }
  else {
    // Run on CPU
    if (gradOutput.dtype() == torch::kFloat32) {
      return mapping_align_generic<float>(gradOutput, indices);
    }
    else if (gradOutput.dtype() == torch::kFloat64) {
      return mapping_align_generic<double>(gradOutput, indices);
    }
    else if (gradOutput.dtype() == torch::kFloat16) {
      return mapping_align_generic<at::Half>(gradOutput, indices);
    }
    else {
      std::cout << "Not implemented for " << gradOutput.dtype() << std::endl;
    }
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mapping_rotate", &mapping_rotate, "ARF rotation via mapping");
  m.def("mapping_align", &mapping_align, "ARF gradient alignment via mapping");
}
