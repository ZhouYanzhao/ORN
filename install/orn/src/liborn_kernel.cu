#include "liborn_kernel.h"

#define FLT_MAX 3.402823466e+38F

template <typename Dtype>
__global__ void MappingRotateKernel(
    const uint32 nthreads, 
    const Dtype* weight_data,
    const uint8* indices_data,
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
        Dtype val = *(weight_data + n);
        for (k = 0; k < nRotation; k++) {
            uint16 index = (uint16)(*(indices_data + l * nRotation + k)) - 1;
            Dtype *target = output_data + i * (nRotation * nInputPlane * nEntry)
                                        + k * (nInputPlane * nEntry)
                                        + j * (nEntry)
                                        + index;
            *target = val;
        }
    }
}

template <typename Dtype>
__global__ void MappingAlignKernel(
    const uint32 nthreads, 
    const Dtype* gradWeight_data,
    const uint8* indices_data,
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
        Dtype *val = weight_data + n;
        *val = 0;
        for (k = 0; k < nRotation; k++) {
            uint16 index = (uint16)(*(indices_data + l * nRotation + k)) - 1;
            Dtype target = *(gradWeight_data + i * (nRotation * nInputPlane * nEntry)
                                             + k * (nInputPlane * nEntry)
                                             + j * (nEntry)
                                             + index);
            *val = *val + target;
        }
    }
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
        const Dtype *src = src_data + i * (nInputPlane * nOrientation * srcEntry)
                                    + j * (nOrientation * srcEntry)
                                    + l * srcEntry;
        const Dtype *elements = indices_data + k * (dstEntry * 8) + m * 8;
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
            const Dtype *src = src_data + i * (nRotation * nInputPlane * nOrientation * dstEntry)
                                    + k * (nInputPlane * nOrientation * dstEntry)
                                    + j * (nOrientation * dstEntry)
                                    + l * dstEntry;
            const Dtype *elements = indices_data + k * (srcEntry * 8) + m * 8;
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
        const Dtype *src = src_data + i * (nRotation * nInputPlane * nOrientation * dstEntry)
                                    + k * (nInputPlane * nOrientation * dstEntry)
                                    + j * (nOrientation * dstEntry)
                                    + m;
        const Dtype *elements = factors_data + + k * (nOrientation * nOrientation) 
                                               + l * nOrientation;
        dst_data[num] = 0;
        for (n = 0; n < nOrientation; n++) {
            dst_data[num] += *(src + n * dstEntry) * elements[n];
        }
    }
}

template <typename Dtype>
__global__ void AlignFeatureKernel(
    const uint32 nthreads, 
    const Dtype* feature_data,
    const uint16 nBatch,
    const uint16 nFeature,
    const uint8 nOrientation,
    uint8* mainDirection_data,
    Dtype* aligned_data) 
{
    CUDA_KERNEL_LOOP(n, nthreads) {
        const uint16 j = n % nFeature;
        const uint16 i = n / nFeature;
        uint8 l;

        uint8 *direction = mainDirection_data + i * nFeature + j;
        Dtype maxVal = -FLT_MAX;
        for (l = 0; l < nOrientation; l++) {
            Dtype val = *(feature_data + i * (nFeature * nOrientation)
                                       + j * (nOrientation)
                                       + l);
            if (val > maxVal) {
                maxVal = val;
                *direction = l;
            }
        }
        for (l = 0; l < nOrientation; l++) {
            Dtype src = *(feature_data + i * (nFeature * nOrientation)
                                       + j * (nOrientation)
                                       + l);
            uint8 alignedIndex = ((l - (uint8)*direction) + nOrientation) % nOrientation;
            Dtype *target = aligned_data + i * (nFeature * nOrientation)
                                         + j * (nOrientation)
                                         + alignedIndex;
            *target = src;
        } 
    }
}

template <typename Dtype>
__global__ void UnAlignFeatureKernel(
    const uint32 nthreads, 
    const Dtype* aligned_data,
    const uint8* mainDirection_data,
    const uint16 nBatch,
    const uint16 nFeature,
    const uint8 nOrientation,
    Dtype* feature_data) 
{
    CUDA_KERNEL_LOOP(n, nthreads) {
        uint8 l;
        const uint16 j = n % nFeature; 
        const uint16 i = n / nFeature;
        const uint8 direction = *(mainDirection_data + i * nFeature + j);
        for (l = 0; l < nOrientation; l++) {
          Dtype src = *(aligned_data + i * (nFeature * nOrientation)
                                     + j * (nOrientation)
                                     + l);
          uint8 alignedIndex = (l + direction) % nOrientation;
          Dtype *target = feature_data + i * (nFeature * nOrientation)
                                       + j * (nOrientation)
                                       + alignedIndex;
          *target = src;
        }
    }
}

#ifdef __cplusplus
extern "C" {
#endif

void kernel_Double_MappingRotate(
    cudaStream_t stream,
    const uint32 count, 
    const double* weight_data,
    const uint8* indices_data,
    const uint16 nInputPlane,
    const uint16 nOutputPlane,
    const uint8 nOrientation,
    const uint8 nRotation,
    const uint16 nEntry,
    double* output_data)
{
    MappingRotateKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>
        (count, weight_data, indices_data, nInputPlane, nOutputPlane, nOrientation, nRotation, nEntry, output_data);
}
void kernel_Float_MappingRotate(
    cudaStream_t stream,
    const uint32 count, 
    const float* weight_data,
    const uint8* indices_data,
    const uint16 nInputPlane,
    const uint16 nOutputPlane,
    const uint8 nOrientation,
    const uint8 nRotation,
    const uint16 nEntry,
    float* output_data)
{
    MappingRotateKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>
        (count, weight_data, indices_data, nInputPlane, nOutputPlane, nOrientation, nRotation, nEntry, output_data);
}

void kernel_Double_MappingAlign(
    cudaStream_t stream,
    const uint32 count, 
    const double* gradWeight_data,
    const uint8* indices_data,
    const uint16 nInputPlane,
    const uint16 nOutputPlane,
    const uint8 nOrientation,
    const uint8 nRotation,
    const uint16 nEntry,
    double* weight_data)
{
    MappingAlignKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>
        (count, gradWeight_data, indices_data, nInputPlane, nOutputPlane, nOrientation, nRotation, nEntry, weight_data);
}
void kernel_Float_MappingAlign(
    cudaStream_t stream,
    const uint32 count, 
    const float* gradWeight_data,
    const uint8* indices_data,
    const uint16 nInputPlane,
    const uint16 nOutputPlane,
    const uint8 nOrientation,
    const uint8 nRotation,
    const uint16 nEntry,
    float* weight_data)
{
    MappingAlignKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>
        (count, gradWeight_data, indices_data, nInputPlane, nOutputPlane, nOrientation, nRotation, nEntry, weight_data);
}

void kernel_Double_AlignFeature(
    cudaStream_t stream,
    const uint32 count,
    const double* feature_data,
    const uint16 nBatch,
    const uint16 nFeature,
    const uint8 nOrientation,
    uint8* mainDirection_data,
    double* aligned_data)
{
    AlignFeatureKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>
      (count, feature_data, nBatch, nFeature, nOrientation, mainDirection_data, aligned_data);
}
void kernel_Float_AlignFeature(
    cudaStream_t stream,
    const uint32 count,
    const float* feature_data,
    const uint16 nBatch,
    const uint16 nFeature,
    const uint8 nOrientation,
    uint8* mainDirection_data,
    float* aligned_data)
{
    AlignFeatureKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>
      (count, feature_data, nBatch, nFeature, nOrientation, mainDirection_data, aligned_data);
}

void kernel_Double_UnAlignFeature(
    cudaStream_t stream,
    const uint32 count, 
    const double* aligned_data,
    const uint8* mainDirection_data,
    const uint16 nBatch,
    const uint16 nFeature,
    const uint8 nOrientation,
    double* feature_data)
{
    UnAlignFeatureKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>
      (count, aligned_data, mainDirection_data, nBatch, nFeature, nOrientation, feature_data);
}
void kernel_Float_UnAlignFeature(
    cudaStream_t stream,
    const uint32 count, 
    const float* aligned_data,
    const uint8* mainDirection_data,
    const uint16 nBatch,
    const uint16 nFeature,
    const uint8 nOrientation,
    float* feature_data)
{
    UnAlignFeatureKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>
      (count, aligned_data, mainDirection_data, nBatch, nFeature, nOrientation, feature_data);
}

#ifdef __cplusplus
}
#endif