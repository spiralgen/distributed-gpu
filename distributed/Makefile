CUDA_PATH = /sw/summit/cuda/10.1.243/
CUDA_INCLUDE = $(CUDA_PATH)
CUDA_LIB = $(CUDA_PATH)/lib64

CC = mpicc
CFLAGS = -std=c99 -I$(CUDA_INCLUDE) -fopenmp


LIBS =  -L$(CUDA_LIB) -lcudart -lcufft -fopenmp
LINKER = $(CC)


OBJS = clean_dist_3dfft.o packing.o debug.o create_plans.o

%.o : %.c
	$(CC) $(CFLAGS) -c $< -o $@

all: $(OBJS)
	$(LINKER) $(OBJS) $(LIBS) -o distributed-3dfft.x


clean:
	rm -f *.o *.x *~ \#*\#


