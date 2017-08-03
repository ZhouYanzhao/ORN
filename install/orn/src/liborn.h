typedef unsigned long uint64;
typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;

int orn_Double_ARF_MappingRotate(
    THDoubleTensor *weight,
    THByteTensor *indices,
    THDoubleTensor *output);
int orn_Float_ARF_MappingRotate(
    THFloatTensor *weight,
    THByteTensor *indices,
    THFloatTensor *output);

int orn_Double_ARF_MappingAlign(
    THDoubleTensor *weight,
    THByteTensor *indices,
    THDoubleTensor *gradWeight);
int orn_Float_ARF_MappingAlign(
    THFloatTensor *weight,
    THByteTensor *indices,
    THFloatTensor *gradWeight);

int orn_Double_RIE_AlignFeature(
    THDoubleTensor *feature,
    THByteTensor *mainDirection,
    THDoubleTensor *aligned,
    const uint8 nOrientation);
int orn_Float_RIE_AlignFeature(
    THFloatTensor *feature,
    THByteTensor *mainDirection,
    THFloatTensor *aligned,
    const uint8 nOrientation);

int orn_Double_RIE_UnAlignFeature(
    THDoubleTensor *feature,
    THByteTensor *mainDirection,
    THDoubleTensor *aligned,
    const uint8 nOrientation);
int orn_Float_RIE_UnAlignFeature(
    THFloatTensor *feature,
    THByteTensor *mainDirection,
    THFloatTensor *aligned,
    const uint8 nOrientation);