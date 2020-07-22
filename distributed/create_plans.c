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

void create_2D1D_local_gpu_plan(int tid, int p, int nGpus,
				int NX, int NY, cufftHandle *stg1_plan, cufftHandle *stg2_plan)
{
  //assumes that GY is a multiple of p

  //stage 1 plan
  //partitions global 2D plane along Y dim
  //each local 2D plane further partitioned along Y dim
  
  //stage 2 plan 
  //partitions global 2D plane along X dim
  //each local 2D plane further partitioned along X dim
  
  cufftResult errCode, result;

  int L1X = NX;
  int L1Y_base = NY / nGpus; int L1Y_offset = 0; int L1Y = L1Y_base;

  if (tid < NY % nGpus){
    L1Y += 1;
    L1Y_offset = tid;
  }
  else{
    L1Y_offset = NY % nGpus;
  }

  int rank = 1;
  int stg1_dims[1] = {L1X};
 
  CUFFT_CHECK(cufftCreate (stg1_plan));
  
  size_t stg1_worksize[1];
  CUFFT_CHECK(cufftMakePlanMany((*stg1_plan), rank, stg1_dims,
				NULL, 1, 1,
				NULL, 1, 1,
				CUFFT_Z2Z, L1Y, stg1_worksize));


  //plan for stage 2
  int L2Y = NY;
  int L2X_base = NX / nGpus; int L2X_offset = 0; int L2X = L2X_base;

  if (tid < NX % nGpus){
    L2X += 1;
    L2X_offset = tid;
  }
  else{
    L2X_offset = NX % nGpus;
  }

  rank = 1;
  int stg2_dims[1] = {L2Y};

  CUFFT_CHECK(cufftCreate (stg2_plan));
  
  size_t stg2_worksize[1];
  //this needs to be updated depending on how data is restructured.
  CUFFT_CHECK(cufftMakePlanMany((*stg2_plan), rank, stg2_dims,
				NULL, 1, 1,
				NULL, 1, 1,
				CUFFT_Z2Z, L2X, stg2_worksize));

  
}


void create_3D1D_local_gpu_plan(int tid, int p, int nGpus, 
				int NX, int NY, int NZ, cufftHandle *stg1_plan, cufftHandle *stg2_plan){
  cufftResult errCode, result;
  
  //stage 1 plan
  int L1X = NX;
  int L1Y = NY;
  int L1Z_base = NZ / nGpus;  int L1Z = L1Z_base;
  int L1Z_offset = 0;
  
  if (tid < NZ % nGpus) {
    L1Z += 1;
    L1Z_offset = tid;
  } else {
    L1Z_offset = NZ % nGpus;
  }

  int rank = 2;
  int stg1_dims[2] = {L1X, L1Y};
 
  CUFFT_CHECK(cufftCreate (stg1_plan));
  
  size_t stg1_worksize[1];
  CUFFT_CHECK(cufftMakePlanMany((*stg1_plan), rank, stg1_dims,
				NULL, 1, 1,
				NULL, 1, 1,
				CUFFT_Z2Z, L1Z , stg1_worksize));

  //stage 2 plan //assumes that GY is a multiple of p
  int N2X = NX;           int L2X = N2X;
  int N2Y = NY / p;       int L2Y_base = N2Y / nGpus; int L2Y = L2Y_base;
  int N2Z = NZ*p;           int L2Z = N2Z;
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
  
  CUFFT_CHECK(cufftCreate (stg2_plan));
  
  size_t stg2_worksize[1];  
  CUFFT_CHECK(cufftMakePlanMany((*stg2_plan), rank, stg2_dims,
				inembed, L2X * L2Y, 1,
				onembed,  1, L2Z,
				CUFFT_Z2Z, L2X * L2Y , stg2_worksize));  
}

void destroy_3D1D_local_gpu_plans(cufftHandle *stg1_plan, cufftHandle *stg2_plan){
  cufftResult errCode, result;
  
  CUFFT_CHECK(cufftDestroy(*stg1_plan));
  CUFFT_CHECK(cufftDestroy(*stg2_plan));
}


void destroy_2D1D_local_gpu_plans(cufftHandle *stg1_plan, cufftHandle *stg2_plan){
  cufftResult errCode, result;
  
  CUFFT_CHECK(cufftDestroy(*stg1_plan));
  CUFFT_CHECK(cufftDestroy(*stg2_plan));
}
