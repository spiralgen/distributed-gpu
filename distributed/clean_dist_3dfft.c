/*
 * FFTX Copyright (c) 2020, The Regents of the University of California, through
 * Lawrence Berkeley National Laboratory (subject to receipt of any required
 * approvals from the U.S. Dept. of Energy), Carnegie Mellon University and
 * SpiralGen, Inc.  All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.
 *
 * NOTICE.  This Software was developed under funding from the U.S. Department of
 * Energy and the U.S. Government  consequently  retains certain rights. As such,
 * the U.S. Government has been granted for itself and others acting on its
 * behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
 * to reproduce, distribute copies to the public, prepare derivative works, and
 * perform publicly and display publicly, and to permit others to do so.
 */

#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <mpi.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <omp.h>

#include "create_plans.h"
#include "debug.h"
#include "packing.h"

#define COMPLEX 2
#define CHECK 1

struct plans_param
{
};


int main(int argc, char *argv[])
{

  int myrank, p, tag=99, err = 0, incorrect = 0;
  MPI_Status status;

  /* Initialize the MPI library */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  const int num_GPU = 6;

  int nGpus = 0;
  cudaError_t cudaError = cudaGetDeviceCount(&nGpus);
  if (cudaError != cudaSuccess) {
    DEBUG_PRINT("%s\n", cudaGetErrorString(cudaError));
  }
  DEBUG_PRINT("expecting %d gpus, found %d gpus\n", num_GPU, nGpus);
  
  int GX, GY, GZ;   //size of overall fft
  int NX, NY, NZ;   //size of fft on the node

  GX = 1024;
  GY = 1024;
  GZ = 1024;

  NX = GX;
  NY = GY;
  NZ = GZ / p;

  DEBUG_PRINT("rank: %d, size: %d\n", myrank, p);

  const size_t buff_size = NX * NY * NZ * 2; //double complex

  //initialize data
  double *host_data_sendbuf = (double *) memalign(64, buff_size * sizeof(double));
  double *host_data_recvbuf = (double *) memalign(64, buff_size * sizeof(double));

  DEBUG_PRINT("init data\n");

  for (size_t k = 0; k < NZ; k++) {
    for (size_t j = 0; j < NY; j++) {
      for (size_t i = 0; i < NX; i++) {
	size_t index = k*NY*NX + j*NX + i;
	double *real = &host_data_sendbuf[2*index];
	double *imag = &host_data_sendbuf[2*index + 1];
	*real = 1.0f * rand() / RAND_MAX;
	*imag = 1.0f * rand() / RAND_MAX;
      }
    }
  }


#pragma omp parallel
  {
    int tid = omp_get_thread_num();

    cufftResult errCode, result;
    
    int L1X, L1Y, L1Z, L1Z_base, L1Z_offset;
    double *local_host_sendbuf, *local_host_recvbuf;

    cudaSetDevice(tid);

    //stage 1 plan
    L1X = NX;
    L1Y = NY;
    L1Z_base = NZ / nGpus;  L1Z = L1Z_base;
    L1Z_offset = 0;

    if (tid < NZ % nGpus) {
      L1Z += 1;
      L1Z_offset = tid;
    } else {
      L1Z_offset = NZ % nGpus;
    }

    local_host_sendbuf = host_data_sendbuf + 2 * L1X * L1Y * (L1Z_base * tid + L1Z_offset);
    local_host_recvbuf = host_data_recvbuf + 2 * L1X * L1Y * (L1Z_base * tid + L1Z_offset);
    
    
    //stage 2 plan //assumes that GY is a multiple of p
    int N2X = GX;           int L2X = N2X;
    int N2Y = GY / p;       int L2Y_base = N2Y / nGpus; int L2Y = L2Y_base;
    int N2Z = GZ;           int L2Z = N2Z;
    int L2Y_offset = 0;

    if (tid < N2Y % nGpus) {
      L2Y += 1;
      L2Y_offset = tid;
    } else {
      L2Y_offset = N2Y % nGpus;
    }
    

    cufftHandle stg1_plan, stg2_plan;
    
    create_3D1D_local_gpu_plan(tid, p, nGpus,
			       NX, NY, NZ,
			       &stg1_plan, &stg2_plan);
    
    //finishing planning

    // allocate device buffers
    cufftDoubleComplex *dev_input;
    cufftDoubleComplex *dev_output;
    int size_req_start = L1X * L1Y * L1Z;
    int size_req_end = L2X * L2Y * L2Z;
    int max_size = (size_req_start > size_req_end ? size_req_start : size_req_end);
    CUDA_CHECK(cudaMalloc((void **) &dev_input, max_size * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc((void **) &dev_output, max_size * sizeof(cufftDoubleComplex)));

    //copy data to device
    //not timed cos we assume GPU to GPU
    CUDA_CHECK(cudaMemcpy(dev_input,
			  local_host_sendbuf,
			  L1X * L1Y * L1Z * sizeof(double) * 2,
			  cudaMemcpyHostToDevice));

    
    #pragma omp barrier

    double start_time = omp_get_wtime();

    // execute on gpus

    //start of 3d fft
    DEBUG_PRINT("exec plan 1\n");
    CUFFT_CHECK(cufftExecZ2Z(stg1_plan, dev_input, dev_output, CUFFT_FORWARD));

    cudaDeviceSynchronize();             // only needed for timing
    double stg1_end = omp_get_wtime();

    //copy data to host
    CUDA_CHECK(cudaMemcpy(
			  local_host_recvbuf,
			  dev_output,
			  L1X * L1Y * L1Z * sizeof(double) * 2,
			  cudaMemcpyDeviceToHost ));

    //pack
    cudaDeviceSynchronize();
    double stg1_gpu_to_host = omp_get_wtime();

    #pragma omp barrier //make sure all gpus are done copying to host

    //pack to prepare for all2all
    double start_pack = omp_get_wtime();
    double *local_src_buf = host_data_recvbuf + (tid * L1Z_base + L1Z_offset) * NX * NY * 2;

    pack_for_all2all(tid,
		     p, nGpus,
		     NX, NY, NZ,
		     host_data_sendbuf, local_src_buf);
    
    #pragma omp barrier    
    double pack_end = omp_get_wtime();    

    //communicate
    #pragma omp single
    {
      MPI_Alltoall(host_data_sendbuf, NZ*NX*NY/p, MPI_C_DOUBLE_COMPLEX,
		   host_data_recvbuf, NZ*NX*NY/p, MPI_C_DOUBLE_COMPLEX,
		   MPI_COMM_WORLD);
    }
    //implicit barrier from omp single and MPI
    double a2a_end = omp_get_wtime();
        
    // swapping dest and src after packing
    double *local_data_sendbuf = host_data_recvbuf + 2 * L2X * L2Z * (L2Y_base * tid + L2Y_offset);
    double *local_data_recvbuf = host_data_sendbuf + 2 * L2X * L2Z * (L2Y_base * tid + L2Y_offset);
    
    //copy data to device
    CUDA_CHECK(cudaMemcpy(dev_input,
			  local_data_recvbuf,
			  L2X * L2Y * L2Z * sizeof(double) * 2,
			  cudaMemcpyHostToDevice));
    
    cudaDeviceSynchronize();  //only needed for timing
    double copy_host_gpu = omp_get_wtime();

    // execute on gpus
    DEBUG_PRINT("exec plan 2\n");
    result = cufftExecZ2Z(stg2_plan, dev_input, dev_output, CUFFT_FORWARD);
    CUFFT_CHECK(result);    
    cudaDeviceSynchronize();

    //end of 3d fft 
    
    double end_time = omp_get_wtime();
    //end of timing
        
    //finalize gpu work
    CUDA_CHECK(cudaFree(dev_input));
    CUDA_CHECK(cudaFree(dev_output));

    destroy_3D1D_local_gpu_plans(&stg1_plan, &stg2_plan);

     
    printf(
	   "%d %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %d\n",
	   myrank, p,
	   start_time,
	   stg1_end,   stg1_gpu_to_host,
	   start_pack, pack_end,
	   a2a_end,
	   copy_host_gpu,
	   end_time,   end_time - start_time,
	   incorrect);
  }   //end parallel region
  
  free(host_data_sendbuf);
  free(host_data_recvbuf);
  
  
  DEBUG_PRINT("finalizing\n");
  MPI_Finalize();
  
 DEBUG_PRINT("done\n");


 return 0;
}
