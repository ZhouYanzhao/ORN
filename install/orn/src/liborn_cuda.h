typedef unsigned long uint64;
typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;

int cuorn_Double_ARF_MappingRotate(
    THCudaDoubleTensor *weight,
    THCudaByteTensor *indices,
    THCudaDoubleTensor *output);
int cuorn_Float_ARF_MappingRotate(
    THCudaTensor *weight,
    THCudaByteTensor *indices,
    THCudaTensor *output);

int cuorn_Double_ARF_MappingAlign(
    THCudaDoubleTensor *weight,
    THCudaByteTensor *indices,
    THCudaDoubleTensor *gradWeight);
int cuorn_Float_ARF_MappingAlign(
    THCudaTensor *weight,
    THCudaByteTensor *indices,
    THCudaTensor *gradWeight);

int cuorn_Double_RIE_AlignFeature(
    THCudaDoubleTensor *feature,
    THCudaByteTensor *mainDirection,
    THCudaDoubleTensor *aligned,
    const uint8 nOrientation);
int cuorn_Float_RIE_AlignFeature(
    THCudaTensor *feature,
    THCudaByteTensor *mainDirection,
    THCudaTensor *aligned,
    const uint8 nOrientation);

int cuorn_Double_RIE_UnAlignFeature(
    THCudaDoubleTensor *feature,
    THCudaByteTensor *mainDirection,
    THCudaDoubleTensor *aligned,
    const uint8 nOrientation);
int cuorn_Float_RIE_UnAlignFeature(
    THCudaTensor *feature,
    THCudaByteTensor *mainDirection,
    THCudaTensor *aligned,
    const uint8 nOrientation);
