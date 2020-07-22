#ifndef _CUDA_ERROR_HELP
#define _CUDA_ERROR_HELP

#define cudaCheckError()                                          \
{                                                                 \
  cudaError_t e = cudaGetLastError();                             \
  if (e != cudaSuccess) {                                         \
    printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,      \
    cudaGetErrorString(e));                                       \
    exit(EXIT_FAILURE);                                           \
  }                                                               \
}

#endif
