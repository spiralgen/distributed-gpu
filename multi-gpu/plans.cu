#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include <omp.h>
#include "cuda_help.h"
#include "plans.h"


void create_slab_pencil_plans(int NX, int NY, int NZ,
		       int LX, int LY, int LZ,
		       cufftHandle *plan1D, cufftHandle *plan2D){
  
  const int dimensions = 2;
  int dims[dimensions] = {NX, NY};    
  
  // Create plans for computing X-Y
  if (cufftPlanMany(plan2D, dimensions, dims,
		    NULL, 1, 1, // in      
		    NULL, 1, 1, // out
		    CUFFT_Z2Z, LZ) != CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT error: Plan creation failed");
    exit(0);
  }
  
  int zdims[1] = {NZ};
  int inembed[1] = {NZ};
  int onembed[1] = {NZ};
  if ( cufftPlanMany(plan1D, 1, zdims,
		     inembed, LX*LY, 1,   // in
		     onembed, 1, NZ,      // out
		     CUFFT_Z2Z, LX*LY
		     ) != CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT error: Plan creation failed");
    exit(0);
  }    
}


void destroy_slab_pencil_plans(cufftHandle *plan1D, cufftHandle *plan2D){
  cufftDestroy(*plan1D);
  cufftDestroy(*plan2D);
}
