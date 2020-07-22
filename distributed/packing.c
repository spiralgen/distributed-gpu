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
#include "packing.h"


void pack_for_all2all(int tid,
		      int p, int nGpus, 
		      int NX, int NY, int NZ,
		      double *host_data_sendbuf, double *local_src_buf)
    {

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
      
      //stage 2 plan
      int N2Y = NY/p; //assumes that NY = GY
      
      //      int N2X = GX;           int L2X = N2X;
      //      int N2Y = GY / p;       int L2Y_base = N2Y / nGpus; int L2Y = L2Y_base;
      //      int N2Z = GZ;           int L2Z = N2Z;
      //      int L2Y_offset = 0;
      
      //      if (tid < N2Y % nGpus) {
      //	L2Y += 1;
      //	L2Y_offset = tid;
      //      } else {
      //	L2Y_offset = N2Y % nGpus;
      //      }
            
      int gpu_base = N2Y/nGpus;
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
    
