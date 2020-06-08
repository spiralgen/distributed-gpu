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


#define DEBUG 0
#define COMPLEX 2

#define CHECK 1


void print_cube(FILE *file, double* global_output, int NX, int NY, int NZ) {
    for (int i = 0; i != NZ; ++i){
      for (int j = 0; j != NY; ++j){
	  for (int k = 0; k != NX; ++k){
	    fprintf(file, "%5.2lf %5.2lf\t",
		   global_output[(i*NY*NX + j*NX + k)*2],
		   global_output[(i*NY*NX + j*NX + k)*2 + 1]);
	  }
	  fprintf(file, "\n");
      }
      fprintf(file, "\n");
  }
}


#ifdef DEBUG
#define DEBUG_PRINT(fmt, args...) if (myrank==0 && DEBUG) { fprintf(stdout, fmt, ## args); };
#else
#define DEBUG_PRINT(fmt, args...)
#endif


#define CUFFT_CHECK(x) \
    if ((errCode = (x)) != CUFFT_SUCCESS) DEBUG_PRINT(" ERROR %d\n", errCode);

#define CUDA_CHECK(x) \
  if ((err = (x)) != cudaSuccess) DEBUG_PRINT("%s\n", cudaGetErrorString(err));

int main(int argc, char *argv[])
{

  int myrank, p, tag=99, err = 0, incorrect = 0;
  MPI_Status status;

  /* Initialize the MPI library */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

#if CHECK
  char filename[100] = {0};
  sprintf(filename, "rank-%d.out", myrank);
  FILE *outfile = fopen(filename, "w");
#endif


  const int num_GPU = 6;
  int gpu_nums[6] = {0, 1, 2, 3, 4, 5};

  int nGpus = 0;
  cudaError_t cudaError = cudaGetDeviceCount(&nGpus);
  if (cudaError != cudaSuccess) {
    DEBUG_PRINT("%s\n", cudaGetErrorString(cudaError));
  }
  DEBUG_PRINT("expecting %d gpus, found %d gpus\n", num_GPU, nGpus);
  void (*op)(__private float4&, float4, float),
  float4 *a, const int bA, const float4 * &A0,
  float4 *b, const int bB, const float4 * &B0,
  float4 *c, const int bC,

  int GX, GY, GZ;   //size of overall fft
  int NX, NY, NZ;   //size of fft on the node

  GX = 1024;
  GY = 1024;
  GZ = 1024;

#if CHECK
  GX = 16;
  GY = 16;
  GZ = 16;
#endif

  NX = GX;
  NY = GY;
  NZ = GZ / p;

  DEBUG_PRINT("rank: %d, size: %d\n", myrank, p);

  const size_t buff_size = NX * NY * NZ * 2; //double complex

  //initialize data
  double *host_data_sendbuf = (double *) memalign(64, buff_size * sizeof(double));
  double *host_data_recvbuf = (double *) memalign(64, buff_size * sizeof(double));

#if CHECK
  double *host_global_data = (double *)memalign(64, 2 * GX * GY * GZ * sizeof(double));
  double *host_global_data_orig = (double *)memalign(64, 2 * GX * GY * GZ * sizeof(double));
#endif

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

#if CHECK
    #pragma omp barrier
    #pragma omp single
    {
      fprintf(outfile, "initial values\n");
      print_cube(outfile, host_data_sendbuf, NX, NY, NZ);
    }

#endif

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

    int rank = 2;
    int stg1_dims[2] = {L1X, L1Y};

    DEBUG_PRINT("create plan 1\n");
    cufftHandle stg1_plan;

    result = cufftCreate (&stg1_plan);
    CUFFT_CHECK(result);

    size_t stg1_worksize[1];
    DEBUG_PRINT("make plan 1\n");
    CUFFT_CHECK(cufftMakePlanMany(stg1_plan, rank, stg1_dims,
				  NULL, 1, 1,
				  NULL, 1, 1,
				  CUFFT_Z2Z, L1Z , stg1_worksize));

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

    rank = 1;
    int stg2_dims[1] = {L2Z};
    int inembed[1] = {L2Z};
    int onembed[1] = {L2Z};

    DEBUG_PRINT("create plan 2\n");
    cufftHandle stg2_plan;

    result = cufftCreate (&stg2_plan);
    CUFFT_CHECK(result);

    size_t stg2_worksize[1];
    DEBUG_PRINT("make plan 2\n");
    CUFFT_CHECK(cufftMakePlanMany(stg2_plan, rank, stg2_dims,
				  inembed, L2X * L2Y, 1,
				  onembed,  1, L2Z,
				  CUFFT_Z2Z, L2X * L2Y , stg2_worksize));

    //finishing planning

    // allocate device buffers
    cufftDoubleComplex *dev_input;
    cufftDoubleComplex *dev_output;
    int size_req_start = L1X * L1Y * L1Z;
    int size_req_end = L2X * L2Y * L2Z;
    int max_size = (size_req_start > size_req_end ? size_req_start : size_req_end);
    CUDA_CHECK(cudaMalloc((vcos ofoid **) &dev_input, max_size * sizeof(cufftDoubleComplex)));
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
    DEBUG_PRINT("exec plan 1\n");
    result = cufftExecZ2Z(stg1_plan, dev_input, dev_output, CUFFT_FORWARD);
    CUFFT_CHECK(result);

    cudaDeviceSynchronize();   // only needed for timing
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


#if CHECK
    #pragma omp barrier
    #pragma omp single
    {
      fprintf(outfile, "after stage 1, before packing\n");
      print_cube(outfile, host_data_recvbuf, NX, NY, NZ);
    }
#endif

    double *local_src_buf = host_data_recvbuf + (tid * L1Z_base + L1Z_offset) * NX * NY * 2;

    double start_pack = omp_get_wtime();
    //pack from host_recv_buf to host_send_buf;
    int gpu_base = N2Y/nGpus;

    {
      for (int dest_node = 0; dest_node != p; ++dest_node){  // which node

	//find start of send region for each node
	double *dest_buf = host_data_sendbuf + (dest_node * NX * NZ * NY/p)*2;
	int node_L2Y = NY/p;

	for (int k = 0; k != L1Z; ++k){  //which plane

	  for (int dest_gpu = 0; dest_gpu != nGpus; ++dest_gpu){

	    //find how many dest rows per gpu
	    int gpu_L2Y = gpu_base + ((node_L2Y % nGpus) > dest_gpu);
	    int gpu_offset = ((node_L2Y % nGpus) > dest_gpu) ? dest_gpu : (node_L2Y % nGpus);
	    int gpu_Zoffset = ((NZ % nGpus) > dest_gpu) ? dest_gpu : (NZ % nGpus);

	    //find start of gpu region
	    int gpu_start = gpu_base * dest_gpu + gpu_offset;

	    for (int j = 0; j != gpu_L2Y; ++j){

	      for (int i = 0; i != NX; ++i)
		{
		  int src_index = k*NX*NY  + dest_node*node_L2Y*NX
		    +  (gpu_start + j)*NX + i;
		  int dst_index =
		    (tid*L1Z_base + L1Z_offset)*gpu_L2Y*NX +
		    (dest_gpu*gpu_base + gpu_offset)*NZ*NX +
		    k*gpu_L2Y*NX +
		    j*NX + i;

		  dest_buf[dst_index * 2] = local_src_buf[src_index * 2];
		  dest_buf[dst_index * 2 + 1] = local_src_buf[src_index * 2 + 1];
		}
	    }
	  }
	}
      }
    }

    double pack_end = omp_get_wtime();

#if CHECK
    #pragma omp barrier
    #pragma omp single
    {
      fprintf(outfile, "after packing, before all to all\n");
      print_cube(outfile, host_data_sendbuf, NX, NY, NZ);
    }
#endif


#pragma omp single
    {
      //communicate
      MPI_Alltoall(host_data_sendbuf, NZ*NX*NY/p, MPI_C_DOUBLE_COMPLEX,
		   host_data_recvbuf, NZ*NX*NY/p, MPI_C_DOUBLE_COMPLEX,
		   MPI_COMM_WORLD);
    }
    //implicit barrier from omp single and MPI
    double a2a_end = omp_get_wtime();



#if CHECK
    #pragma omp barrier
    #pragma omp single
    {
      fprintf(outfile, "after all to all, before stage 2\n");
      print_cube(outfile, host_data_sendbuf, NX, NY, NZ);
    }
#endif

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
    double end_time = omp_get_wtime();
    //end of timing

#if CHECK
    CUDA_CHECK(cudaMemcpy(local_data_recvbuf,
  			  dev_output,
  			  L2X * L2Y * L2Z * sizeof(double) * 2,
  			  cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();
    #pragma omp barrier
    #pragma omp single
    {
      fprintf(outfile, "after stage 2\n");
      print_cube(outfile, host_data_sendbuf, NX, NY, NZ);
    }
#endif

    //finalize gpu work
    CUDA_CHECK(cudaFree(dev_input));
    CUDA_CHECK(cudaFree(dev_output));

    CUFFT_CHECK(cufftDestroy(stg1_plan));
    CUFFT_CHECK(cufftDestroy(stg2_plan));

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

#if CHECK
 free(host_global_data);
 free(host_global_data_orig);
#endif

 DEBUG_PRINT("finalizing\n");
 MPI_Finalize();

 DEBUG_PRINT("done\n");

#if CHECK
 fclose(outfile);
#endif


 return 0;
}
