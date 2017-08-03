typedef unsigned long uint64;
typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// Use 1024 threads per block, which requires cuda sm_2x or above
const int CUDA_NUM_THREADS = 1024;
 
// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

#ifdef __cplusplus
extern "C" {
#endif

void kernel_Double_MappingRotate(
    cudaStream_t stream,
    const uint32 nthreads, 
    const double* weight_data,
    const uint8* indices_data,
    const uint16 nInputPlane,
    const uint16 nOutputPlane,
    const uint8 nOrientation,
    const uint8 nRotation,
    const uint16 nEntry,
    double* output_data);
void kernel_Float_MappingRotate(
    cudaStream_t stream,
    const uint32 nthreads, 
    const float* weight_data,
    const uint8* indices_data,
    const uint16 nInputPlane,
    const uint16 nOutputPlane,
    const uint8 nOrientation,
    const uint8 nRotation,
    const uint16 nEntry,
    float* output_data);

void kernel_Double_MappingAlign(
    cudaStream_t stream,
    const uint32 nthreads, 
    const double* gradWeight_data,
    const uint8* indices_data,
    const uint16 nInputPlane,
    const uint16 nOutputPlane,
    const uint8 nOrientation,
    const uint8 nRotation,
    const uint16 nEntry,
    double* weight_data);
void kernel_Float_MappingAlign(
    cudaStream_t stream,
    const uint32 nthreads, 
    const float* gradWeight_data,
    const uint8* indices_data,
    const uint16 nInputPlane,
    const uint16 nOutputPlane,
    const uint8 nOrientation,
    const uint8 nRotation,
    const uint16 nEntry,
    float* weight_data);

void kernel_Double_AlignFeature(
    cudaStream_t stream,
    const uint32 count,
    const double* feature_data,
    const uint16 nBatch,
    const uint16 nFeature,
    const uint8 nOrientation,
    uint8* mainDirection_data,
    double* aligned_data);
void kernel_Float_AlignFeature(
    cudaStream_t stream,
    const uint32 count,
    const float* feature_data,
    const uint16 nBatch,
    const uint16 nFeature,
    const uint8 nOrientation,
    uint8* mainDirection_data,
    float* aligned_data);

void kernel_Double_UnAlignFeature(
    cudaStream_t stream,
    const uint32 count, 
    const double* aligned_data,
    const uint8* mainDirection_data,
    const uint16 nBatch,
    const uint16 nFeature,
    const uint8 nOrientation,
    double* feature_data);
void kernel_Float_UnAlignFeature(
    cudaStream_t stream,
    const uint32 count, 
    const float* aligned_data,
    const uint8* mainDirection_data,
    const uint16 nBatch,
    const uint16 nFeature,
    const uint8 nOrientation,
    float* feature_data);

#ifdef __cplusplus
}
#endif