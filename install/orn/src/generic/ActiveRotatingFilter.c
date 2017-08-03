#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/ActiveRotatingFilter.c"
#else

int orn_(ARF_MappingRotate)(
    THTensor *weight,
    THByteTensor *indices,
    THTensor *output)
{
    THArgCheck(weight->nDimension == 5, 1, "only supports a batch of ARFs.");
    const uint16 nOutputPlane = weight->size[0];
    const uint16 nInputPlane = weight->size[1];
    const uint8 nOrientation = weight->size[2];
    const uint8 kH = weight->size[3];
    const uint8 kW = weight->size[4];
    const uint8 nRotation = indices->size[3];

    THTensor_(resize4d)(output, nOutputPlane * nRotation, nInputPlane * nOrientation, kH, kW);

    real *weightData = THTensor_(data)(weight);
    uint8 *indicesData = THByteTensor_data(indices);
    real *outputData = THTensor_(data)(output);

    const uint16 nEntry = nOrientation * kH * kW;
    uint16 i, j, l;
    uint8 k;

    #pragma omp parallel for private(i, j, l, k)
    for (i = 0; i < nOutputPlane; i++) {
        for (j = 0; j < nInputPlane; j++) {
            for (l = 0; l < nEntry; l++) {
                real val = *(weightData++);
                for (k = 0; k < nRotation; k++) {
                    uint8 index = *(indicesData + l * nRotation + k) - 1;
                    real *target = outputData + i * (nRotation * nInputPlane * nEntry)
                                              + k * (nInputPlane * nEntry)
                                              + j * (nEntry)
                                              + index;
                    *target = val;
                }
            }
        }
    }

    return 1;
}

int orn_(ARF_MappingAlign)(
    THTensor *weight,
    THByteTensor *indices,
    THTensor *gradWeight)
{
    const uint8 nOrientation = indices->size[0];
    const uint8 kH = indices->size[1];
    const uint8 kW = indices->size[2];
    const uint8 nRotation = indices->size[3];
    const uint16 nOutputPlane = gradWeight->size[0] / nRotation;
    const uint16 nInputPlane = gradWeight->size[1] / nOrientation;

    THTensor_(resize5d)(weight, nOutputPlane, nInputPlane, nOrientation, kH, kW);

    real *weightData = THTensor_(data)(weight);
    uint8 *indicesData = THByteTensor_data(indices);
    real *gradWeightData = THTensor_(data)(gradWeight);

    const uint16 nEntry = nOrientation * kH * kW;
    uint16 i, j, l;
    uint8 k;

    #pragma omp parallel for private(i, j, l, k)
    for (i = 0; i < nOutputPlane; i++) {
        for (j = 0; j < nInputPlane; j++) {
            for (l = 0; l < nEntry; l++) {
                real *val = weightData++;
                *val = 0;
                for (k = 0; k < nRotation; k++) {
                    uint8 index = *(indicesData + l * nRotation + k) - 1;
                    real *target = gradWeightData + i * (nRotation * nInputPlane * nEntry)
                                                  + k * (nInputPlane * nEntry)
                                                  + j * (nEntry)
                                                  + index;
                    *val += *target;
                }
            }
        }
    }

    return 1;
}
#endif
