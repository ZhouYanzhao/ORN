#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/RotationInvariantEncoding.c"
#else

int orn_(RIE_AlignFeature)(
    THTensor *feature,
    THByteTensor *mainDirection,
    THTensor *aligned,
    const uint8 nOrientation)
{
    THArgCheck(feature->nDimension == 4, 1, "only supports batch mode.");

    const uint16 nBatch = feature->size[0];
    const uint16 nChannel = feature->size[1];
    const uint16 nFeature = nChannel / nOrientation;
    THArgCheck(feature->size[2] == 1 && feature->size[3] == 1, 1, "mH x mW should be 1x1.");

    THByteTensor_resize2d(mainDirection, nBatch, nFeature);
    THTensor_(resizeAs)(aligned, feature);

    real *feature_data = THTensor_(data)(feature);
    uint8 *mainDirection_data = THByteTensor_data(mainDirection);
    real *aligned_data = THTensor_(data)(aligned);

    uint16 i;
    uint16 j;
    uint8 l;

    #pragma omp parallel for private(i, j, l)
    for (i = 0; i < nBatch; i++) {
        for (j = 0; j < nFeature; j++) {
            uint8 *direction = mainDirection_data + i * nFeature + j;
            real maxVal = -THInf;
            for (l = 0; l < nOrientation; l++) {
                real val = *(feature_data + i * (nFeature * nOrientation)
                                          + j * (nOrientation)
                                          + l);
                if (val > maxVal) {
                    maxVal = val;
                    *direction = l;
                }
            }
            for (l = 0; l < nOrientation; l++) {
                real src = *(feature_data + i * (nFeature * nOrientation)
                                          + j * (nOrientation)
                                          + l);
                uint8 alignedIndex = (l - (uint8)*direction + nOrientation) % nOrientation;
                real *target = aligned_data + i * (nFeature * nOrientation)
                                            + j * (nOrientation)
                                            + alignedIndex;
                *target = src;
            }
        }
    }

    return 1;
}

int orn_(RIE_UnAlignFeature)(
    THTensor *feature,
    THByteTensor *mainDirection,
    THTensor *aligned,
    const uint8 nOrientation)
{
    const uint16 nBatch = mainDirection->size[0];
    const uint16 nFeature = mainDirection->size[1];

    THTensor_(resizeAs)(feature, aligned);

    real *feature_data = THTensor_(data)(feature);
    uint8 *mainDirection_data = THByteTensor_data(mainDirection);
    real *aligned_data = THTensor_(data)(aligned);

    uint16 i;
    uint16 j;
    uint8 l;

    #pragma omp parallel for private(i, j, l)
    for (i = 0; i < nBatch; i++) {
        for (j = 0; j < nFeature; j++) {
            uint8 direction = *(mainDirection_data + i * nFeature + j);
            for (l = 0; l < nOrientation; l++) {
                real src = *(aligned_data + i * (nFeature * nOrientation)
                                          + j * (nOrientation)
                                          + l);
                uint8 alignedIndex = (l + direction) % nOrientation;
                real *target = feature_data + i * (nFeature * nOrientation)
                                            + j * (nOrientation)
                                            + alignedIndex;
                *target = src;
            }
        }
    }
    
    return 1;
}

#endif
