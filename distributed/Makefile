CUDA_PATH = /usr/local/cuda-10.1/
CUDA_INCLUDE = $(CUDA_PATH)/include
CUDA_LIB = $(CUDA_PATH)/lib64

CC = mpicc
CFLAGS = -std=c99 -I$(CUDA_INCLUDE)


LIBS =  -L$(CUDA_LIB) -lcudart -lcufft -fopenmp
LINKER = $(CC)


OBJS = clean_dist_3dfft.o packing.o debug.o create_plans.o

%.o : %.c
	$(CC) $(CFLAGS) -c $< -o $@

all: $(OBJS)
	$(LINKER) $(OBJS) $(LIBS) -o distributed-3dfft.x


clean:
	rm -f *.o *.x *~ \#*\#
#distributed-3dfft.x: clean_dist_3dfft.c
#	mpicc -I/usr/local/cuda-10.1/include -L -lcudart -lcufft -fopenmp -DOMP_NUM_THREADS=6 clean_dist_3dfft.c -o distributed-3dfft.x
