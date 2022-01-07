#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include <omp.h>
#include "cuda_help.h"
#include "packing.h"
#include "plans.h"
#include "gpu_comms.h"

int main(int argc, char **argv) {

  int mult = atoi(argv[1]);

  int numGPUs;

  cudaError_t cudaStatus;

  cudaGetDeviceCount(&numGPUs);
  cudaCheckError();

  int p = numGPUs;
  omp_set_num_threads(p);  // create as many CPU threads as there are CUDA devices

  // allocate device buffers
  cufftDoubleComplex**     sendbufs = (cufftDoubleComplex **) malloc(p * sizeof(cufftDoubleComplex *));
  cufftDoubleComplex**     recvbufs = (cufftDoubleComplex **) malloc(p * sizeof(cufftDoubleComplex *));

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    cudaSetDevice(tid);
    
    size_t N = 6*mult;
    size_t NX, NY, NZ;
    NX = NY = NZ = N;

    // enable peer to peer.
    cudaStream_t streams[NUM_STREAMS];    
    init_GPU_peer(p, tid, streams);    

    cudaMalloc(&sendbufs[tid], mult * N * N * sizeof(cufftDoubleComplex));
    cudaCheckError();
    cudaMalloc(&recvbufs[tid], mult * N * N * sizeof(cufftDoubleComplex));
    cudaCheckError();

    #pragma omp barrier

    // PLAN FFTS
    int LX = NX;
    int LY, LZ;

    size_t send_size[p];
    for (int i = 0; i != p; ++i) {
      LY = NY/p;
      if (NY % p > i) {
        LY += 1;
      }

      LZ = NZ/p;
      if (NZ % p > i) {
        LZ += 1;
      }

      send_size[i] = NX * LY * LZ;
    }

    size_t offsets[p];
    offsets[0] = 0;
    for (int i = 1; i != p; ++i) {
      offsets[i] = offsets[i-1] + send_size[i-1];
    }

    cufftHandle plan2D, plan1D;
    create_slab_pencil_plans(NX, NY, NZ,
			     LX, LY, LZ,
			     &plan1D, &plan2D);
    
#pragma omp barrier  //even start time
    double start_2d_compute = omp_get_wtime(); // seconds
	
    //execute 2D FFTs, batched in the z dimension
    if (
	cufftExecZ2Z(
		     plan2D,         // cufftHandle plan
		     sendbufs[tid],  // cufftComplex *idata
		     recvbufs[tid],  // cufftComplex *odata
		     CUFFT_FORWARD   // int direction
		     ) != CUFFT_SUCCESS) {
      fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
      exit(0);
    }
    
    //wait for FFT to stop before calling all2all
    cudaStatus = cudaDeviceSynchronize();
    double end_2d_compute = omp_get_wtime();
    
    if (cudaStatus != cudaSuccess) {
          fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
          exit(0);
    }
    
    
    double start_packing = omp_get_wtime();
    
    /***********************************************************************/
        dim3 thread_block(min(LX,1024),1,1);
        dim3 compute_grid(1,LY,LZ);
        for (int i = 0; i < p; i++) {
          pack<<<compute_grid, thread_block, 0, streams[i % NUM_STREAMS]>>>(
            // index to partion y along the front slab
            recvbufs[tid] + i*LY*LX,  // in
            NX, // in_x,
            NY, // in_y,
            LZ, // in_z,
            LX, // dim_x,
            LY, // dim_y,
            LZ, // dim_z,
            sendbufs[tid] + offsets[i]  // out
          );
        }

        for (int s = 0; s < NUM_STREAMS; s++) {
          cudaStreamSynchronize(streams[s]);
          cudaCheckError();
        }


        double end_packing = omp_get_wtime();
        cudaCheckError();


        #pragma omp barrier

        //all to all
        // perform A2A
        double start_transfer = omp_get_wtime();

        for (int i = 0; i < p; ++i)
        {
          int dst = (tid + i) % p;
          cudaMemcpyPeerAsync(
            recvbufs[dst] + offsets[tid],    // dest pointer.
            dst,                             // dest device.
            sendbufs[tid] +  offsets[dst],   // src pointer.
            tid,                             // src device.
            send_size[dst] * sizeof(cufftDoubleComplex), // num of bytes
            streams[i % NUM_STREAMS]         // stream to perform the copy on.
          );
          cudaCheckError();
        }

        for (int s = 0; s < NUM_STREAMS; s++) {
          cudaStreamSynchronize(streams[s]);
          cudaCheckError();
        }
	/***********************************************************************/	

        double end_transfer = omp_get_wtime();
        cudaCheckError();


        #pragma omp barrier    //have to wait until all GPUs have finish sending data
        double start_1d_compute = omp_get_wtime();

        if (cufftExecZ2Z(plan1D, sendbufs[tid], recvbufs[tid], CUFFT_FORWARD) !=
        CUFFT_SUCCESS) {
          fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
          exit(0);
        }

        //wait for FFT to stop before calling all2all
        cudaStatus = cudaDeviceSynchronize();
        double end_1d_compute = omp_get_wtime();

        if (cudaStatus != cudaSuccess) {
          fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
          exit(0);
        }

        #pragma omp barrier

        // seconds
        double end_time = omp_get_wtime();

        end_time -= start_2d_compute;
        end_1d_compute -= start_2d_compute;
        start_1d_compute -= start_2d_compute;
        end_transfer -= start_2d_compute;
        start_transfer -= start_2d_compute;
        end_packing -= start_2d_compute;
        start_packing -= start_2d_compute;
        end_2d_compute -= start_2d_compute;
        start_2d_compute -= start_2d_compute;
        printf("%zu,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
	       N,
	       tid,
	       start_2d_compute,
	       end_2d_compute,
	       start_packing,
	       end_packing,
	       start_transfer,
	       end_transfer,
	       start_1d_compute,
	       end_1d_compute,
	       end_time
       );

      // cleanup.
      finalize_GPU_peer(streams);
      
      cudaFree(sendbufs[tid]);
      cudaCheckError();
      cudaFree(recvbufs[tid]);
      cudaCheckError();

      destroy_slab_pencil_plans(&plan1D, &plan2D);
      
    } // end parallel region
    free(sendbufs);
    free(recvbufs);    
  }
