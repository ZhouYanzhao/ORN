#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/ActiveRotatingFilter.cu"
#else

int cuorn_(ARF_MappingRotate)(
    THCTensor *weight,
    THCudaByteTensor *indices,
    THCTensor *output)
{
    THCUNN_assertSameGPU(state, 3, weight, indices, output);
    THArgCheck(weight->nDimension == 5, 1, "only supports a batch of ARFs.");
    const uint16 nOutputPlane = weight->size[0];
    const uint16 nInputPlane = weight->size[1];
    const uint8 nOrientation = weight->size[2];
    const uint8 kH = weight->size[3];
    const uint8 kW = weight->size[4];
    const uint8 nRotation = indices->size[3];

    THCTensor_(resize4d)(state, output, nOutputPlane * nRotation, nInputPlane * nOrientation, kH, kW);

    real *weightData = THCTensor_(data)(state, weight);
    uint8 *indicesData = THCudaByteTensor_data(state, indices);
    real *outputData = THCTensor_(data)(state, output);

    const uint16 nEntry = nOrientation * kH * kW;
    const uint32 count = nOutputPlane * nInputPlane * nEntry;

    kernel_(MappingRotate)(
        THCState_getCurrentStream(state),
        count, 
        weightData, 
        indicesData, 
        nInputPlane, 
        nOutputPlane, 
        nOrientation, 
        nRotation, 
        nEntry, 
        outputData);
    THCudaCheck(cudaGetLastError());

    return 1;
}

int cuorn_(ARF_MappingAlign)(
    THCTensor *weight,
    THCudaByteTensor *indices,
    THCTensor *gradWeight)
{
    THCUNN_assertSameGPU(state, 3, weight, indices, gradWeight);
    const uint8 nOrientation = indices->size[0];
    const uint8 kH = indices->size[1];
    const uint8 kW = indices->size[2];
    const uint8 nRotation = indices->size[3];
    const uint16 nOutputPlane = gradWeight->size[0] / nRotation;
    const uint16 nInputPlane = gradWeight->size[1] / nOrientation;

    THCTensor_(resize5d)(state, weight, nOutputPlane, nInputPlane, nOrientation, kH, kW);

    real *weightData = THCTensor_(data)(state, weight);
    uint8 *indicesData = THCudaByteTensor_data(state, indices);
    real *gradWeightData = THCTensor_(data)(state, gradWeight);

    const uint16 nEntry = nOrientation * kH * kW;
    const uint32 count = nOutputPlane * nInputPlane * nEntry;

    kernel_(MappingAlign)(
        THCState_getCurrentStream(state),
        count, 
        gradWeightData, 
        indicesData, 
        nInputPlane, 
        nOutputPlane, 
        nOrientation, 
        nRotation, 
        nEntry, 
        weightData);
    THCudaCheck(cudaGetLastError());

    return 1;
}

#endif