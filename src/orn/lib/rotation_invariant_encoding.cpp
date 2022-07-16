#include <vector>
#include <torch/extension.h>


#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define INF 2.0e+10F

// CUDA declarations
std::vector<torch::Tensor> align_feature_generic_cuda(torch::Tensor input, uint8_t nOrientation);
torch::Tensor unalign_feature_generic_cuda(torch::Tensor gradOutput, torch::Tensor mainDirection, uint8_t nOrientation);

// CPU declarations
template <typename scalar_t>
std::vector<torch::Tensor> align_feature_generic(torch::Tensor input, uint8_t nOrientation) {
  const auto nBatch = input.size(0);
  const auto nChannel = input.size(1);
  const auto nFeature = nChannel / nOrientation;
  const auto kH = input.size(2);
  const auto kW = input.size(3);
  AT_ASSERTM((kH == 1) && (kW == 1), "kH x kW should be 1x1.");

  auto mainDirection = torch::empty({nBatch, nFeature}, input.options().dtype(torch::kUInt8));
  input = input.view({nBatch, nFeature, nOrientation});
  auto input_accessor = input.accessor<scalar_t, 3>();
  auto output = torch::empty({nBatch, nFeature, nOrientation}, input.options());
  auto output_accessor = output.accessor<scalar_t, 3>();
  auto mainDirection_accessor = mainDirection.accessor<uint8_t, 2>();
  
  for (auto i = 0; i < nBatch; i++) {
    for (auto j = 0; j < nFeature; j++) {
      // Find main directions
      uint8_t l;
      scalar_t maxVal = -INF;
      for (l = 0; l < nOrientation; l++) {
        const scalar_t val = input_accessor[i][j][l];
        if (val > maxVal) {
            maxVal = val;
            mainDirection_accessor[i][j] = l;
        }
      }
      // Align features
      for (l = 0; l < nOrientation; l++) {
        const uint8_t alignedIndex = (l - mainDirection_accessor[i][j] + nOrientation) % nOrientation;
        output_accessor[i][j][alignedIndex] = input_accessor[i][j][l];
      }
    }
  }
  return {output.view({nBatch, nChannel, 1, 1}), mainDirection};
}

template <typename scalar_t>
torch::Tensor unalign_feature_generic(torch::Tensor gradOutput, torch::Tensor mainDirection, uint8_t nOrientation) {
  const auto nBatch = gradOutput.size(0);
  const auto nChannel = gradOutput.size(1);
  const auto nFeature = nChannel / nOrientation;
  const auto kH = gradOutput.size(2);
  const auto kW = gradOutput.size(3);
  AT_ASSERTM((kH == 1) && (kW == 1), "kH x kW should be 1x1.");

  gradOutput = gradOutput.view({nBatch, nFeature, nOrientation});
  auto gradOutput_accessor = gradOutput.accessor<scalar_t, 3>();
  auto gradInput = torch::empty({nBatch, nFeature, nOrientation}, gradOutput.options());
  auto gradInput_accessor = gradInput.accessor<scalar_t, 3>();
  auto mainDirection_accessor = mainDirection.accessor<uint8_t, 2>();
  
  for (auto i = 0; i < nBatch; i++) {
    for (auto j = 0; j < nFeature; j++) {
      // Unalign gradients
      for (auto l = 0; l < nOrientation; l++) {
        const uint8_t alignedIndex = (l + mainDirection_accessor[i][j]) % nOrientation;
        gradInput_accessor[i][j][alignedIndex] = gradOutput_accessor[i][j][l];
      }
    }
  }
  return gradInput.view({nBatch, nChannel, 1, 1});
}

// C++ interface
std::vector<torch::Tensor> align_feature(torch::Tensor input, uint8_t nOrientation) {
  // Check inputs
  CHECK_CONTIGUOUS(input);

  if (input.device().is_cuda()) {
    // Run on CUDA
    return align_feature_generic_cuda(input, nOrientation);
  }
  else {
    // Run on CPU
    if (input.dtype() == torch::kFloat32) {
      return align_feature_generic<float>(input, nOrientation);
    }
    else if (input.dtype() == torch::kFloat64) {
      return align_feature_generic<double>(input, nOrientation);
    }
    else if (input.dtype() == torch::kFloat16) {
      return align_feature_generic<at::Half>(input, nOrientation);
    }
    else {
      std::cout << "Not implemented for " << input.dtype() << std::endl;
    }
  }
}

torch::Tensor unalign_feature(torch::Tensor gradOutput, torch::Tensor mainDirection, uint8_t nOrientation) {
  // Check inputs
  CHECK_CONTIGUOUS(gradOutput);
  CHECK_CONTIGUOUS(mainDirection);

  if ((gradOutput.device().is_cuda()) || (mainDirection.device().is_cuda())) {
    // Run on CUDA
    return unalign_feature_generic_cuda(gradOutput, mainDirection, nOrientation);
  }
  else {
    // Run on CPU
    if (gradOutput.dtype() == torch::kFloat32) {
      return unalign_feature_generic<float>(gradOutput, mainDirection, nOrientation);
    }
    else if (gradOutput.dtype() == torch::kFloat64) {
      return unalign_feature_generic<double>(gradOutput, mainDirection, nOrientation);
    }
    else if (gradOutput.dtype() == torch::kFloat16) {
      return unalign_feature_generic<at::Half>(gradOutput, mainDirection, nOrientation);
    }
    else {
      std::cout << "Not implemented for " << gradOutput.dtype() << std::endl;
    }
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("align_feature", &align_feature, "Rotation invariant encoding");
  m.def("unalign_feature", &unalign_feature, "Rotation invariant encoding backprop");
}
