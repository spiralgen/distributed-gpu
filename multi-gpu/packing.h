#include <cufft.h>
#include <omp.h>

// from a cube of size in_x by in_y by in_z,
// partition the cube along the y axis via offset_y and dim_y,
// copying to a contiguous buffer out
__global__ void pack(
  cufftDoubleComplex * __restrict__ in,
  int in_x,
  int in_y,
  int in_z,
  int dim_x,
  int dim_y,
  int dim_z,
  cufftDoubleComplex  * __restrict__ out
) {
  for (
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    k < dim_z;
    k += gridDim.z*blockDim.z
  ) {
    for (
      int j = blockIdx.y*blockDim.y + threadIdx.y;
      j < dim_y;
      j += gridDim.y*blockDim.y
    ) {
      for (
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        i < dim_x;
        i += gridDim.x*blockDim.x
      ) {
        out[
        k*(dim_x * dim_z) +
        j*(dim_x) +
        i
        ] = in[
        k*(in_x * in_z) +
        j*(in_x) +
        i
        ];
      }
    }
  }
}

