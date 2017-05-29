template <typename Dtype>
__global__ void MappingRotateKernel(
    const uint32 nthreads, 
    const Dtype* weight_data,
    const Dtype* indices_data,
    const uint16 nInputPlane,
    const uint16 nOutputPlane,
    const uint8 nOrientation,
    const uint8 nRotation,
    const uint16 nEntry,
    Dtype* output_data) 
{
    CUDA_KERNEL_LOOP(n, nthreads) {
        uint16 l = n % nEntry;
        uint16 j = (n / nEntry) % nInputPlane;
        uint16 i = n / nEntry / nInputPlane;
        uint8 k;
        float val = *(weight_data + n);
        for (k = 0; k < nRotation; k++) {
            uint16 index = (uint16)(*(indices_data + l * nRotation + k)) - 1;
            float *target = output_data + i * (nRotation * nInputPlane * nEntry)
                                        + k * (nInputPlane * nEntry)
                                        + j * (nEntry)
                                        + index;
            *target = val;
        }
    }
}

static int cuorn_ARF_MappingRotate(lua_State *L)
{
    THCState *state = getCutorchState(L);
    THCudaTensor *weight = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
    THCudaTensor *indices = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
    THCudaTensor *output = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
    const uint8 kW = lua_tonumber(L, 5);
    const uint8 kH = lua_tonumber(L, 6);
    const uint16 nInputPlane = lua_tonumber(L, 7);
    const uint16 nOutputPlane = lua_tonumber(L, 8);
    const uint8 nOrientation = lua_tonumber(L, 9);
    const uint8 nRotation = lua_tonumber(L, 10);
    THCUNN_assertSameGPU(state, 3, weight, indices, output);

    float *weight_data = THCudaTensor_data(state, weight);
    float *indices_data = THCudaTensor_data(state, indices);
    float *output_data = THCudaTensor_data(state, output);

    const uint16 nEntry = nOrientation * kH * kW;
    const uint32 count = nOutputPlane * nInputPlane * nEntry;

    MappingRotateKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
        (count, weight_data, indices_data, nInputPlane, nOutputPlane, nOrientation, nRotation, nEntry, output_data);
    THCudaCheck(cudaGetLastError());

    return 0;
}

template <typename Dtype>
__global__ void MappingAlignKernel(
    const uint32 nthreads, 
    const Dtype* gradWeight_data,
    const Dtype* indices_data,
    const uint16 nInputPlane,
    const uint16 nOutputPlane,
    const uint8 nOrientation,
    const uint8 nRotation,
    const uint16 nEntry,
    Dtype* weight_data) 
{
    CUDA_KERNEL_LOOP(n, nthreads) {
        uint16 l = n % nEntry;
        uint16 j = (n / nEntry) % nInputPlane;
        uint16 i = n / nEntry / nInputPlane;
        uint8 k;
        float *val = weight_data + n;
        for (k = 0; k < nRotation; k++) {
            uint16 index = (uint16)(*(indices_data + l * nRotation + k)) - 1;
            float target = *(gradWeight_data + i * (nRotation * nInputPlane * nEntry)
                                             + k * (nInputPlane * nEntry)
                                             + j * (nEntry)
                                             + index);
            *val = *val + target;
        }
    }
}

static int cuorn_ARF_MappingAlign(lua_State *L)
{
    THCState *state = getCutorchState(L);
    THCudaTensor *weight = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
    THCudaTensor *indices = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
    THCudaTensor *gradWeight = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
    const uint8 kW = lua_tonumber(L, 5);
    const uint8 kH = lua_tonumber(L, 6);
    const uint16 nInputPlane = lua_tonumber(L, 7);
    const uint16 nOutputPlane = lua_tonumber(L, 8);
    const uint8 nOrientation = lua_tonumber(L, 9);
    const uint8 nRotation = lua_tonumber(L, 10);
    THCUNN_assertSameGPU(state, 3, weight, indices, gradWeight);

    float *weight_data = THCudaTensor_data(state, weight);
    float *indices_data = THCudaTensor_data(state, indices);
    float *gradWeight_data = THCudaTensor_data(state, gradWeight);

    const uint16 nEntry = nOrientation * kH * kW;
    const uint32 count = nOutputPlane * nInputPlane * nEntry;

    MappingAlignKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
        (count, gradWeight_data, indices_data, nInputPlane, nOutputPlane, nOrientation, nRotation, nEntry, weight_data);
    THCudaCheck(cudaGetLastError());

    return 0;
}

template <typename Dtype>
__global__ void RotateKernel(
    const uint64 nthreads, 
    const Dtype* src_data,
    const Dtype* indices_data,
    const uint16 nInputPlane,
    const uint16 nOutputPlane,
    const uint8 nOrientation,
    const uint8 nRotation,
    const uint16 srcEntry,
    const uint16 dstEntry,
    Dtype* dst_data) 
{
    CUDA_KERNEL_LOOP(num, nthreads) {
        const uint16 m = num % dstEntry;
        const uint8 l = (num / dstEntry) % nOrientation;
        const uint16 j = (num / dstEntry / nOrientation) % nInputPlane;
        const uint8 k = (num / dstEntry / nOrientation / nInputPlane) % nRotation;
        const uint16 i = (num / dstEntry / nOrientation / nInputPlane / nRotation);
        const float *src = src_data + i * (nInputPlane * nOrientation * srcEntry)
                                    + j * (nOrientation * srcEntry)
                                    + l * srcEntry;
        const float *elements = indices_data + k * (dstEntry * 8) + m * 8;
        dst_data[num] = *(src + (uint8)elements[1]) * elements[0]
                      + *(src + (uint8)elements[3]) * elements[2]
                      + *(src + (uint8)elements[5]) * elements[4]
                      + *(src + (uint8)elements[7]) * elements[6];
    }
}

template <typename Dtype>
__global__ void AlignKernel(
    const uint64 nthreads, 
    const Dtype* src_data,
    const Dtype* indices_data,
    const uint16 nInputPlane,
    const uint16 nOutputPlane,
    const uint8 nOrientation,
    const uint8 nRotation,
    const uint16 srcEntry,
    const uint16 dstEntry,
    Dtype* dst_data) 
{
    CUDA_KERNEL_LOOP(num, nthreads) {
        uint8 k;
        const uint16 m = num % srcEntry;
        const uint8 l = (num / srcEntry) % nOrientation;
        const uint16 j = (num / srcEntry / nOrientation) % nInputPlane;
        const uint16 i = (num / srcEntry / nOrientation / nInputPlane);
        for (k = 0; k < nRotation; k++) {
            const float *src = src_data + i * (nRotation * nInputPlane * nOrientation * dstEntry)
                                    + k * (nInputPlane * nOrientation * dstEntry)
                                    + j * (nOrientation * dstEntry)
                                    + l * dstEntry;
            const float *elements = indices_data + k * (srcEntry * 8) + m * 8;
            dst_data[num] += *(src + (long)elements[1]) * elements[0]
                          + *(src + (long)elements[3]) * elements[2]
                          + *(src + (long)elements[5]) * elements[4]
                          + *(src + (long)elements[7]) * elements[6];
        }
    }
}

template <typename Dtype>
__global__ void SpinKernel(
    const uint64 nthreads, 
    const Dtype* src_data,
    const Dtype* factors_data,
    const uint16 nInputPlane,
    const uint16 nOutputPlane,
    const uint8 nOrientation,
    const uint8 nRotation,
    const uint16 srcEntry,
    const uint16 dstEntry,
    Dtype* dst_data) 
{
    CUDA_KERNEL_LOOP(num, nthreads) {
        uint8 n;
        const uint16 m = num % dstEntry;
        const uint8 l = (num / dstEntry) % nOrientation;
        const uint16 j = (num / dstEntry / nOrientation) % nInputPlane;
        const uint8 k = (num / dstEntry / nOrientation / nInputPlane) % nRotation;
        const uint16 i = (num / dstEntry / nOrientation / nInputPlane / nRotation);
        const float *src = src_data + i * (nRotation * nInputPlane * nOrientation * dstEntry)
                                    + k * (nInputPlane * nOrientation * dstEntry)
                                    + j * (nOrientation * dstEntry)
                                    + m;
        const float *elements = factors_data + + k * (nOrientation * nOrientation) 
                                               + l * nOrientation;
        dst_data[num] = 0;
        for (n = 0; n < nOrientation; n++) {
            dst_data[num] += *(src + n * dstEntry) * elements[n];
        }
    }
}


static int cuorn_ARF_Rotate(lua_State *L)
{
    THCState *state = getCutorchState(L);
    THCudaTensor *weight = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
    THCudaTensor *indices = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
    THCudaTensor *factors = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
    THCudaTensor *weightBuffer = (THCudaTensor*)luaT_checkudata(L, 5, "torch.CudaTensor");
    THCudaTensor *buffer = (THCudaTensor*)luaT_checkudata(L, 6, "torch.CudaTensor");
    const uint8 srcW = lua_tonumber(L, 7);
    const uint8 srcH = lua_tonumber(L, 8);
    const uint8 dstW = lua_tonumber(L, 9);
    const uint8 dstH = lua_tonumber(L, 10);
    const uint16 nInputPlane = lua_tonumber(L, 11);
    const uint16 nOutputPlane = lua_tonumber(L, 12);
    const uint8 nOrientation = lua_tonumber(L, 13);
    const uint8 nRotation = lua_tonumber(L, 14);
    THCUNN_assertSameGPU(state, 5, weight, indices, factors, weightBuffer, buffer);

    float *weightData = THCudaTensor_data(state, weight);
    float *indicesData = THCudaTensor_data(state, indices);
    float *factorsData = THCudaTensor_data(state, factors);
    float *bufferData = THCudaTensor_data(state, buffer);
    float *weightBufferData = THCudaTensor_data(state, weightBuffer);

    float *target;
    const uint16 srcEntry = srcH * srcW;
    const uint16 dstEntry = dstH * dstW;
    const uint64 count = nOutputPlane * nRotation * nInputPlane * nOrientation * dstEntry;

    target = (nOrientation == 1) ? weightBufferData : bufferData;

    RotateKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
        (count, weightData, indicesData, nInputPlane, nOutputPlane, nOrientation, nRotation, srcEntry, dstEntry, target);
    THCudaCheck(cudaGetLastError());

    if (nOrientation == 1) 
        return 0;

    SpinKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
        (count, bufferData, factorsData, nInputPlane, nOutputPlane, nOrientation, nRotation, srcEntry, dstEntry, weightBufferData);
    THCudaCheck(cudaGetLastError());
    return 0;
}

static int cuorn_ARF_Align(lua_State *L)
{
    THCState *state = getCutorchState(L);
    THCudaTensor *gradWeight = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
    THCudaTensor *indices = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
    THCudaTensor *factors = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
    THCudaTensor *buffer = (THCudaTensor*)luaT_checkudata(L, 5, "torch.CudaTensor");
    THCudaTensor *gradWeightBuffer = (THCudaTensor*)luaT_checkudata(L, 6, "torch.CudaTensor");
    const uint8 srcW = lua_tonumber(L, 7);
    const uint8 srcH = lua_tonumber(L, 8);
    const uint8 dstW = lua_tonumber(L, 9);
    const uint8 dstH = lua_tonumber(L, 10);
    const uint16 nInputPlane = lua_tonumber(L, 11);
    const uint16 nOutputPlane = lua_tonumber(L, 12);
    const uint8 nOrientation = lua_tonumber(L, 13);
    const uint8 nRotation = lua_tonumber(L, 14);
    THCUNN_assertSameGPU(state, 5, gradWeight, indices, factors, buffer, gradWeightBuffer);

    float *gradWeightData = THCudaTensor_data(state, gradWeight);
    float *indicesData = THCudaTensor_data(state, indices);
    float *factorsData = THCudaTensor_data(state, factors);
    float *gradWeightBufferData = THCudaTensor_data(state, gradWeightBuffer);
    float *bufferData = THCudaTensor_data(state, buffer);

    const uint16 srcEntry = srcH * srcW;
    const uint16 dstEntry = dstH * dstW;

    uint64 count;

    if (nOrientation > 1) {
        count = nOutputPlane * nRotation * nInputPlane * nOrientation * dstEntry;
        SpinKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
            (count, gradWeightBufferData, factorsData, nInputPlane, nOutputPlane, nOrientation, nRotation, srcEntry, dstEntry, bufferData);
        THCudaCheck(cudaGetLastError());
    }
    else {
        bufferData = gradWeightBufferData;
    }

    count = nOutputPlane * nInputPlane * nOrientation * srcEntry;
    AlignKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
        (count, bufferData, indicesData, nInputPlane, nOutputPlane, nOrientation, nRotation, srcEntry, dstEntry, gradWeightData);
    THCudaCheck(cudaGetLastError());

    return 0;
}

static const struct luaL_Reg cuorn_ARF__ [] = {
    {"ARF_MappingRotate", cuorn_ARF_MappingRotate},
    {"ARF_MappingAlign", cuorn_ARF_MappingAlign},
    {"ARF_Rotate", cuorn_ARF_Rotate},
    {"ARF_Align", cuorn_ARF_Align},
    {NULL, NULL}
};

static void cuorn_ARF_init(lua_State *L)
{
    luaT_pushmetatable(L, "torch.CudaTensor");
    luaT_registeratname(L, cuorn_ARF__, "orn");
    lua_pop(L,1);
}