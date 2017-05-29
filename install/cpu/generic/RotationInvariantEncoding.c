#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/RotationInvariantEncoding.c"
#else

static int orn_(RIE_AlignFeature)(lua_State *L)
{
    THTensor *feature = luaT_checkudata(L, 2, torch_Tensor);
    THTensor *mainDirection = luaT_checkudata(L, 3, torch_Tensor);
    THTensor *aligned = luaT_checkudata(L, 4, torch_Tensor);
    const uint16 nBatch = lua_tonumber(L, 5);
    const uint16 nFeature = lua_tonumber(L, 6);
    const uint8 nOrientation = lua_tonumber(L, 7);
    uint16 i;
    uint16 j;
    uint8 l;

    real *feature_data = THTensor_(data)(feature);
    real *mainDirection_data = THTensor_(data)(mainDirection);
    real *aligned_data = THTensor_(data)(aligned);

#pragma omp parallel for private(i)
    for (i = 0; i < nBatch; i++) {
        for (j = 0; j < nFeature; j++) {
            real *direction = mainDirection_data + i * nFeature + j;
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

    return 0;
}

static int orn_(RIE_UnAlignFeature)(lua_State *L)
{
    THTensor *feature = luaT_checkudata(L, 2, torch_Tensor);
    THTensor *mainDirection = luaT_checkudata(L, 3, torch_Tensor);
    THTensor *aligned = luaT_checkudata(L, 4, torch_Tensor);
    const uint16 nBatch = lua_tonumber(L, 5);
    const uint16 nFeature = lua_tonumber(L, 6);
    const uint8 nOrientation = lua_tonumber(L, 7);
    uint16 i;
    uint16 j;
    uint8 l;

    real *feature_data = THTensor_(data)(feature);
    real *mainDirection_data = THTensor_(data)(mainDirection);
    real *aligned_data = THTensor_(data)(aligned);

#pragma omp parallel for private(i)
    for (i = 0; i < nBatch; i++) {
        for (j = 0; j < nFeature; j++) {
            uint8 direction = (uint8)*(mainDirection_data + i * nFeature + j);
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
    
    return 0;
}

static const struct luaL_Reg orn_(RIE__) [] = {
    {"RIE_AlignFeature", orn_(RIE_AlignFeature)},
    {"RIE_UnAlignFeature", orn_(RIE_UnAlignFeature)},
    {NULL, NULL}
};

static void orn_(RIE_init)(lua_State *L)
{
    luaT_pushmetatable(L, torch_Tensor);
    luaT_registeratname(L, orn_(RIE__), "orn");
    lua_pop(L,1);
}

#endif
