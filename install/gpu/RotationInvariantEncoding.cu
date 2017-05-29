template <typename Dtype>
__global__ void AlignFeatureKernel(
    const uint32 nthreads, 
    const Dtype* feature_data,
    const uint16 nBatch,
    const uint16 nFeature,
    const uint8 nOrientation,
    Dtype* mainDirection_data,
    Dtype* aligned_data) 
{
    CUDA_KERNEL_LOOP(n, nthreads) {
        const uint16 j = n % nFeature;
        const uint16 i = n / nFeature;
        uint8 l;

        float *direction = mainDirection_data + i * nFeature + j;
        float maxVal = -FLT_MAX;
        for (l = 0; l < nOrientation; l++) {
            float val = *(feature_data + i * (nFeature * nOrientation)
                                       + j * (nOrientation)
                                       + l);
            if (val > maxVal) {
                maxVal = val;
                *direction = l;
            }
        }
        for (l = 0; l < nOrientation; l++) {
            float src = *(feature_data + i * (nFeature * nOrientation)
                                       + j * (nOrientation)
                                       + l);
            uint8 alignedIndex = ((l - (uint8)*direction) + nOrientation) % nOrientation;
            float *target = aligned_data + i * (nFeature * nOrientation)
                                         + j * (nOrientation)
                                         + alignedIndex;
            *target = src;
        } 
    }
}

static int cuorn_RIE_AlignFeature(lua_State *L)
{
    THCState *state = getCutorchState(L);
    THCudaTensor *feature = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
    THCudaTensor *mainDirection = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
    THCudaTensor *aligned = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
    const uint16 nBatch = lua_tonumber(L, 5);
    const uint16 nFeature = lua_tonumber(L, 6);
    const uint8 nOrientation = lua_tonumber(L, 7);
    THCUNN_assertSameGPU(state, 3, feature, mainDirection, aligned);

    float *feature_data = THCudaTensor_data(state, feature);
    float *mainDirection_data = THCudaTensor_data(state, mainDirection);
    float *aligned_data = THCudaTensor_data(state, aligned);

    const uint32 count = nBatch * nFeature;

    AlignFeatureKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
      (count, feature_data, nBatch, nFeature, nOrientation, mainDirection_data, aligned_data);
    THCudaCheck(cudaGetLastError());

    return 0;
}

template <typename Dtype>
__global__ void UnAlignFeatureKernel(
    const uint32 nthreads, 
    const Dtype* aligned_data,
    const Dtype* mainDirection_data,
    const uint16 nBatch,
    const uint16 nFeature,
    const uint8 nOrientation,
    Dtype* feature_data) 
{
    CUDA_KERNEL_LOOP(n, nthreads) {
        uint8 l;
        const uint16 j = n % nFeature; 
        const uint16 i = n / nFeature;
        const uint8 direction = (uint8)*(mainDirection_data + i * nFeature + j);
        for (l = 0; l < nOrientation; l++) {
          float src = *(aligned_data + i * (nFeature * nOrientation)
                                     + j * (nOrientation)
                                     + l);
          uint8 alignedIndex = (l + direction) % nOrientation;
          float *target = feature_data + i * (nFeature * nOrientation)
                                       + j * (nOrientation)
                                       + alignedIndex;
          *target = src;
        }
    }
}

static int cuorn_RIE_UnAlignFeature(lua_State *L)
{
    THCState *state = getCutorchState(L);
    THCudaTensor *feature = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
    THCudaTensor *mainDirection = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
    THCudaTensor *aligned = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
    const uint16 nBatch = lua_tonumber(L, 5);
    const uint16 nFeature = lua_tonumber(L, 6);
    const uint8 nOrientation = lua_tonumber(L, 7);
    THCUNN_assertSameGPU(state, 3, feature, mainDirection, aligned);

    float *feature_data = THCudaTensor_data(state, feature);
    float *mainDirection_data = THCudaTensor_data(state, mainDirection);
    float *aligned_data = THCudaTensor_data(state, aligned);

    const uint32 count = nBatch * nFeature;

    UnAlignFeatureKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
      (count, aligned_data, mainDirection_data, nBatch, nFeature, nOrientation, feature_data);
    THCudaCheck(cudaGetLastError());
    
    return 0;
}

static const struct luaL_Reg cuorn_RIE__ [] = {
    {"RIE_AlignFeature", cuorn_RIE_AlignFeature},
    {"RIE_UnAlignFeature", cuorn_RIE_UnAlignFeature},
    {NULL, NULL}
};

static void cuorn_RIE_init(lua_State *L)
{
    luaT_pushmetatable(L, "torch.CudaTensor");
    luaT_registeratname(L, cuorn_RIE__, "orn");
    lua_pop(L,1);
}
