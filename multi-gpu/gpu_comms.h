#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include <omp.h>
#include "cuda_help.h"

#ifndef NUM_STREAMS
#define NUM_STREAMS 6
#endif

void init_GPU_peer(int p, int tid, cudaStream_t *streams);

void finalize_GPU_peer(cudaStream_t *streams);
