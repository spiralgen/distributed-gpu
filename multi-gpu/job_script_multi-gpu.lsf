#!/bin/bash
### Begin BSUB Options
#BSUB -P  PROJ_ID ## use your own project number 
#BSUB -J TEST
#BSUB -W 0:10
#BSUB -nnodes 1

module load cuda

export OMP_NUM_THREADS=6
jsrun --nrs 1 --cpu_per_rs 6 --gpu_per_rs 6 --rs_per_host 1 --np 1 --latency_priority CPU-CPU --launch_distribution cyclic --bind rs ./packed_fft.x 32
jsrun --nrs 1 --cpu_per_rs 6 --gpu_per_rs 6 --rs_per_host 1 --np 1 --latency_priority CPU-CPU --launch_distribution cyclic --bind rs ./fused_packed.x 32

