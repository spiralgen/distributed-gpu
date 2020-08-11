# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda
HOST_COMPILER ?= g++
NVCC          := nvcc -ccbin $(HOST_COMPILER)

# Common includes and paths for CUDA
INCLUDES  := 
LIBRARIES :=

LIBRARIES += -lcufft
LIBRARIES += -Xcompiler -fopenmp

all:
	$(NVCC) $(INCLUDES) -o fused_packed.x fused_packed.cu plans.cu gpu_comms.cu $(LIBRARIES)
	$(NVCC) $(INCLUDES) -o packed_fft.x packed_fft.cu plans.cu gpu_comms.cu $(LIBRARIES)

clean:
	rm -f *~ *.x \#*\#
