#include <TH/TH.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "liborn.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch., Real, Tensor)
#define orn_(NAME) TH_CONCAT_4(orn_, Real, _, NAME)

#include "generic/ActiveRotatingFilter.c"
#include "THGenerateFloatTypes.h"

#include "generic/RotationInvariantEncoding.c"
#include "THGenerateFloatTypes.h"