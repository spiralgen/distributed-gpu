#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include <omp.h>
#include "cuda_help.h"
#include "gpu_comms.h"


void init_GPU_peer(int p, int tid, cudaStream_t *streams){
  // enable peer to peer.
  for (int dd = 0; dd < p; dd++) {
    int access = 0;
    cudaDeviceCanAccessPeer(&access, tid, dd);
    if (access){
        cudaDeviceEnablePeerAccess(dd, 0);
        cudaCheckError();
    }
  }

  for (int s = 0; s < NUM_STREAMS; s++) {
    cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking);
    cudaCheckError();
  }

}

void finalize_GPU_peer(cudaStream_t *streams){
  for (int s = 0; s < NUM_STREAMS; s++) {
    cudaStreamDestroy(streams[s]);
    cudaCheckError();
  }  
}
