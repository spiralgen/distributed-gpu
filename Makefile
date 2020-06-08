distributed-3dfft.x: distributed-3dfft.c
	mpicc -lcudart -lcufft -fopenmp -L/sw/summit/cuda/10.1.243/lib64/ -DOMP_NUM_THREADS=6 distributed-3dfft.c -o distributed-3dfft.x
