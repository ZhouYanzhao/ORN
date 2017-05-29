#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/ActiveRotatingFilter.c"
#else

static int orn_(ARF_MappingRotate)(lua_State *L)
{
    THTensor *weight = luaT_checkudata(L, 2, torch_Tensor);
    THTensor *indices = luaT_checkudata(L, 3, torch_Tensor);
    THTensor *output = luaT_checkudata(L, 4, torch_Tensor);
    const uint8 kW = lua_tonumber(L, 5);
    const uint8 kH = lua_tonumber(L, 6);
    const uint16 nInputPlane = lua_tonumber(L, 7);
    const uint16 nOutputPlane = lua_tonumber(L, 8);
    const uint8 nOrientation = lua_tonumber(L, 9);
    const uint8 nRotation = lua_tonumber(L, 10);

    real *weightData = THTensor_(data)(weight);
    real *indicesData = THTensor_(data)(indices);
    real *outputData = THTensor_(data)(output);

    const uint16 nEntry = nOrientation * kH * kW;
    uint16 i, j, l;
    uint8 k;

    #pragma omp parallel for private(i)
    for (i = 0; i < nOutputPlane; i++) {
        for (j = 0; j < nInputPlane; j++) {
            for (l = 0; l < nEntry; l++) {
                real val = *(weightData++);
                for (k = 0; k < nRotation; k++) {
                    uint16 index = (uint16)(*(indicesData + l * nRotation + k)) - 1;
                    real *target = outputData + i * (nRotation * nInputPlane * nEntry)
                                              + k * (nInputPlane * nEntry)
                                              + j * (nEntry)
                                              + index;
                    *target = val;
                }
            }
        }
    }

    return 0;
}

static int orn_(ARF_MappingAlign)(lua_State *L)
{
    THTensor *weight = luaT_checkudata(L, 2, torch_Tensor);
    THTensor *indices = luaT_checkudata(L, 3, torch_Tensor);
    THTensor *gradWeight = luaT_checkudata(L, 4, torch_Tensor);
    const uint8 kW = lua_tonumber(L, 5);
    const uint8 kH = lua_tonumber(L, 6);
    const uint16 nInputPlane = lua_tonumber(L, 7);
    const uint16 nOutputPlane = lua_tonumber(L, 8);
    const uint8 nOrientation = lua_tonumber(L, 9);
    const uint8 nRotation = lua_tonumber(L, 10);

    real *weightData = THTensor_(data)(weight);
    real *indicesData = THTensor_(data)(indices);
    real *gradWeightData = THTensor_(data)(gradWeight);

    const uint16 nEntry = nOrientation * kH * kW;
    uint16 i, j, l;
    uint8 k;


    #pragma omp parallel for private(i)
    for (i = 0; i < nOutputPlane; i++) {
        for (j = 0; j < nInputPlane; j++) {
            for (l = 0; l < nEntry; l++) {
                real *val = weightData++;
                for (k = 0; k < nRotation; k++) {
                    uint16 index = (uint16)(*(indicesData + l * nRotation + k)) - 1;
                    real *target = gradWeightData + i * (nRotation * nInputPlane * nEntry)
                                                  + k * (nInputPlane * nEntry)
                                                  + j * (nEntry)
                                                  + index;
                    *val += *target;
                }
            }
        }
    }

    return 0;
}

static int orn_(ARF_Rotate)(lua_State *L)
{
    THTensor *weight = luaT_checkudata(L, 2, torch_Tensor);
    THTensor *indices = luaT_checkudata(L, 3, torch_Tensor);
    THTensor *factors = luaT_checkudata(L, 4, torch_Tensor);
    THTensor *weightBuffer = luaT_checkudata(L, 5, torch_Tensor);
    THTensor *buffer = luaT_checkudata(L, 6, torch_Tensor);
    const uint8 srcW = lua_tonumber(L, 7);
    const uint8 srcH = lua_tonumber(L, 8);
    const uint8 dstW = lua_tonumber(L, 9);
    const uint8 dstH = lua_tonumber(L, 10);
    const uint16 nInputPlane = lua_tonumber(L, 11);
    const uint16 nOutputPlane = lua_tonumber(L, 12);
    const uint8 nOrientation = lua_tonumber(L, 13);
    const uint8 nRotation = lua_tonumber(L, 14);

    real *weightData = THTensor_(data)(weight);
    real *indicesData = THTensor_(data)(indices);
    real *factorsData = THTensor_(data)(factors);
    real *bufferData = THTensor_(data)(buffer);
    real *weightBufferData = THTensor_(data)(weightBuffer);

    real *src;
    real *target;
    real *elements;
    const uint16 srcEntry = srcH * srcW;
    const uint16 dstEntry = dstH * dstW;
    uint16 i, j, m;
    uint8 l, n, k;

    target = (nOrientation == 1) ? weightBufferData : bufferData;

    #pragma omp parallel for private(i)
    for (i = 0; i < nOutputPlane; i++) {
        for (k = 0; k < nRotation; k++) {
            for (j = 0; j < nInputPlane; j++) {
                for (l = 0; l < nOrientation; l++) {
                    src = weightData + i * (nInputPlane * nOrientation * srcEntry)
                                     + j * (nOrientation * srcEntry)
                                     + l * srcEntry;
                    for (m = 0; m < dstEntry; m++) {
                        elements = indicesData + k * (dstEntry * 8) + m * 8;
                        *(target++) = *(src + (uint8)elements[1]) * elements[0]
                                    + *(src + (uint8)elements[3]) * elements[2]
                                    + *(src + (uint8)elements[5]) * elements[4]
                                    + *(src + (uint8)elements[7]) * elements[6];
                    }
                }
            }       
        }
    }

    if (nOrientation == 1) 
        return 0;

    target = weightBufferData;

    #pragma omp parallel for private(i)
    for (i = 0; i < nOutputPlane; i++) {
        for (k = 0; k < nRotation; k++) {
            for (j = 0; j < nInputPlane; j++) {
                for (l = 0; l < nOrientation; l++) {
                    for (m = 0; m < dstEntry; m++) {
                        src = bufferData + i * (nRotation * nInputPlane * nOrientation * dstEntry)
                                         + k * (nInputPlane * nOrientation * dstEntry)
                                         + j * (nOrientation * dstEntry)
                                         + m;
                        elements = factorsData + k * (nOrientation * nOrientation) 
                                               + l * nOrientation;
                        *target = 0.0f;
                        for (n = 0; n < nOrientation; n++) {
                            *target += *(src + n * dstEntry) * elements[n];
                        }
                        target++;
                    }
                }
            }       
        }
    }

    return 0;
}

static int orn_(ARF_Align)(lua_State *L)
{
    THTensor *gradWeight = luaT_checkudata(L, 2, torch_Tensor);
    THTensor *indices = luaT_checkudata(L, 3, torch_Tensor);
    THTensor *factors = luaT_checkudata(L, 4, torch_Tensor);
    THTensor *buffer = luaT_checkudata(L, 5, torch_Tensor);
    THTensor *gradWeightBuffer = luaT_checkudata(L, 6, torch_Tensor);
    const uint8 srcW = lua_tonumber(L, 7);
    const uint8 srcH = lua_tonumber(L, 8);
    const uint8 dstW = lua_tonumber(L, 9);
    const uint8 dstH = lua_tonumber(L, 10);
    const uint16 nInputPlane = lua_tonumber(L, 11);
    const uint16 nOutputPlane = lua_tonumber(L, 12);
    const uint8 nOrientation = lua_tonumber(L, 13);
    const uint8 nRotation = lua_tonumber(L, 14);

    real *gradWeightData = THTensor_(data)(gradWeight);
    real *indicesData = THTensor_(data)(indices);
    real *factorsData = THTensor_(data)(factors);
    real *gradWeightBufferData = THTensor_(data)(gradWeightBuffer);
    real *bufferData = THTensor_(data)(buffer);

    real *src;
    real *target;
    real *elements;
    const uint8 srcEntry = srcH * srcW;
    const uint8 dstEntry = dstH * dstW;
    uint16 i, j, m;
    uint8 l, n, k;

    if (nOrientation > 1) {
        target = bufferData;
        #pragma omp parallel for private(i)
        for (i = 0; i < nOutputPlane; i++) {
            for (k = 0; k < nRotation; k++) {
                for (j = 0; j < nInputPlane; j++) {
                    for (l = 0; l < nOrientation; l++) {
                        for (m = 0; m < dstEntry; m++) {
                            src = gradWeightBufferData + i * (nRotation * nInputPlane * nOrientation * dstEntry)
                                                       + k * (nInputPlane * nOrientation * dstEntry)
                                                       + j * (nOrientation * dstEntry)
                                                       + m;
                            elements = factorsData + k * (nOrientation * nOrientation) 
                                                   + l * nOrientation;
                            *target = 0.0f;
                            for (n = 0; n < nOrientation; n++) {
                                *target += *(src + n * dstEntry) * elements[n];
                            }
                            target++;
                        }
                    }
                }       
            }
        }
    }
    else {
        bufferData = gradWeightBufferData;
    }

    target = gradWeightData;
    #pragma omp parallel for private(i)
    for (i = 0; i < nOutputPlane; i++) {
        for (j = 0; j < nInputPlane; j++) {
            for (l = 0; l < nOrientation; l++) {
                for (m = 0; m < srcEntry; m++) {
                    for (k = 0; k < nRotation; k++) {
                        src = bufferData + i * (nRotation * nInputPlane * nOrientation * dstEntry)
                                         + k * (nInputPlane * nOrientation * dstEntry)
                                         + j * (nOrientation * dstEntry)
                                         + l * dstEntry;
                        elements = indicesData + k * (srcEntry * 8) + m * 8;
                        *target += *(src + (uint8)elements[1]) * elements[0]
                                + *(src + (uint8)elements[3]) * elements[2]
                                + *(src + (uint8)elements[5]) * elements[4]
                                + *(src + (uint8)elements[7]) * elements[6];
                    }
                    target++;
                }
            }
        }       
    }

    return 0;
}

static const struct luaL_Reg orn_(ARF__) [] = {
    {"ARF_MappingRotate", orn_(ARF_MappingRotate)},
    {"ARF_MappingAlign", orn_(ARF_MappingAlign)},
    {"ARF_Rotate", orn_(ARF_Rotate)},
    {"ARF_Align", orn_(ARF_Align)},
    {NULL, NULL}
};

static void orn_(ARF_init)(lua_State *L)
{
    luaT_pushmetatable(L, torch_Tensor);
    luaT_registeratname(L, orn_(ARF__), "orn");
    lua_pop(L,1);
}

#endif
