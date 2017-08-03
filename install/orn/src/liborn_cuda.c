#include <THC/THC.h>
#include "liborn_kernel.h"
#include "liborn_cuda.h"

#define cuorn_(NAME) TH_CONCAT_4(cuorn_, Real, _, NAME)
#define kernel_(NAME) TH_CONCAT_4(kernel_, Real, _, NAME)
#define THCUNN_assertSameGPU(...) THAssertMsg(THCudaTensor_checkGPU(__VA_ARGS__), \
  "Some of weight/gradient/input tensors are located on different GPUs. Please move them to a single one.")

// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;

#include "generic/RotationInvariantEncoding.cu"
#include <THC/THCGenerateFloatType.h>

#include "generic/RotationInvariantEncoding.cu"
#include <THC/THCGenerateDoubleType.h>

#include "generic/ActiveRotatingFilter.cu"
#include <THC/THCGenerateFloatType.h>

#include "generic/ActiveRotatingFilter.cu"
#include <THC/THCGenerateDoubleType.h>