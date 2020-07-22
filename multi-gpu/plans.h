#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include <omp.h>
#include "cuda_help.h"

void create_slab_pencil_plans(int NX, int NY, int NZ,
			      int LX, int LY, int LZ,
			      cufftHandle *plan1D, cufftHandle *plan2D);


void destroy_slab_pencil_plans(cufftHandle *plan1D, cufftHandle *plan2D);
