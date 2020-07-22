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
#include <cufftXt.h>


#define DEBUG 0

//#ifdef DEBUG
//#define DEBUG_PRINT(fmt, args...) if (myrank==0 && DEBUG) { fprintf(stdout, fmt, ## args); };
//#else
#define DEBUG_PRINT(fmt, args...)
//#endif


#define CUFFT_CHECK(x) \
    if ((errCode = (x)) != CUFFT_SUCCESS) DEBUG_PRINT(" ERROR %d\n", errCode);

#define CUDA_CHECK(x) \
  if ((err = (x)) != cudaSuccess) DEBUG_PRINT("%s\n", cudaGetErrorString(err));


void print_cube(FILE *file, double* global_output, int NX, int NY, int NZ);
