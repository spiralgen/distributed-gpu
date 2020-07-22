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

void create_3D1D_local_gpu_plan(int tid, int p, int nGpus,
				int NX, int NY, int NZ,
				cufftHandle *stg1_plan, cufftHandle *stg2_plan);
void create_2D1D_local_gpu_plan(int tid, int p, int nGpus,
				int NX, int NY, 
				cufftHandle *stg1_plan, cufftHandle *stg2_plan);

void destroy_3D1D_local_gpu_plans(cufftHandle *stg1_plan, cufftHandle *stg2_plan);
void destroy_2D1D_local_gpu_plans(cufftHandle *stg1_plan, cufftHandle *stg2_plan);
