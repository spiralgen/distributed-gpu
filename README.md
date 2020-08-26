Distributed GPU Backend for FFTX 
---
This is the (planned) distributed GPU library backend for FFTX. This
library will support both multi-GPUs on a single node, and multi-GPUs
across different nodes in a distributed system.

[FFTX](http://spiral.net/doc/fftx/introduction.html) is the exascale follow-on to the FFTW open source discrete FFT
package for executing the Fast Fourier Transform as well as
higher-level operations composed of linear operations combined with
DFT transforms. Though backwards compatible with FFTW, this is an
entirely new work developed as a cooperative effort between Lawrence
Berkeley National Laboratory, Carnegie Mellon University, and
SpiralGen, Inc.

---

## Parallel Multi-dimensional FFT Communication Experiments

The performance of parallel Fast Fourier Transforms (FFTs) is tied to
the characteristics of the communication mechanism that implements the
all-to-all communication collective, a required communication
pattern. As the all-to-all can be implemented with many different
algorithms, it is necessary to determine either 
1. Identify the characteristics of the all-to-all collective communication, or
2. Understand the underlying communication mechanism so as to identify the 
appropriate way for implementing the all-to-all collective communication.

These codes in this repository are part of a series of experiements
designed to identify the characteristics of the communication
mechanism between multiple GPUs and across multiple nodes on a
distributed system. These experiment test the all-to-all communication
within the context of a 3D-FFT that has been partitioned in a
slab-pencil (1D) fashion. 

Two sets of experiments, under different conditions, are designed:

1. [Multi-GPU experiments](#multi-gpu-experiments)
2. [Distributed experiments](#distributed-experiments).


**Note: The final distribution of the 3D-FFT is not in natural format. They are
interleaved across all processors.**

---

## Multi-GPU experiments

This set of multi-GPU experiments is designed to identify the
trade-off between two approaches of implementing the packing for the
all-to-all communications required between stages.  Specifically, the
experiment is designed to identify the startup cost of the
communication between GPUs to better hide the cost of the packing. By
running the same algorithm/configuration with increasing FFT sizes,
the trade off between two different methods of performing the packing
can be identified.

The all-to-all communication between GPUs is implemented as
asynchronous memory copies betweeen peer GPUs
(``cudaMemcpyPeerAsync``) with multiple streams. The all-to-all
implementation supports **exactly** 6 GPUs.

The two approaches that are tested are:
1. Packing the entire 3D FFT before the all-to-all (``packed_fft``).
2. Performing packing as part of the all-to-all (``fused_packed``).

### Compilation
Before building, update `CUDA_PATH` in the `Makefile` to the CUDA install path you'd like to use.
```
make                    #creates 2 executables packed_fft.x and fused_packed.x
```

### Execution
```
./packed_fft.x 32       #packs data before communication on a (32 x 6)^3 FFT on 6 GPUs
./fused_packed.x 32     #fuses the packing with the communication on a (32 x 6)^3 FFT on 6 GPUs
```		
### Running on Summit
A job script is included for running on Summit. **You will need to update the script with a valid project number.** 

To run an experiment with multiple different sizes on Summit, replace the job with the following commands:
```
for n in 32 48 64 80 96 112 128 134 160 171 176
  do
    for run in {1..10}
      do
        jsrun --nrs 1 --cpu_per_rs 6 --gpu_per_rs 6 --rs_per_host 1 --np 1 --latency_priority CPU-CPU --launch_distribution cyclic --bind rs <<executable>> $n
    done
done
``` 

### Outputs
The output of each of the experiment is in Common Separated Value (CSV) format. The 
fields for ``fused_packed`` AND ``packed_fft`` are as follows:
```distributed-3dfft.x
size, gpu_id, start_2d_compute, end_2d_compute,  start_distributed-3dfft.xtransfer,  end_transfer,  
start_1d_compute, end_1d_compute, end_time
```
and 
```
size, gpu_id, start_2d_compute, end_2d_compute,  start_packing, end_packing, 
start_transfer,  end_transfer,  start_1d_compute, end_1d_compute, end_time
```
respectively.

---

## Distributed Experiments

This set of experiements measures the performance characteristics of
the MPI all-to-all implementation over increasing number of nodes. The
performance of any collective communication (including the all-to-all)
is dependent on the underlying assumptions inherent in the
implementation. These assumptions includes the algorithm(s) being
used, the performance of the algorithms as the number of nodes
increases. Local computation is assumed to be executed by all 6 GPUs
on a Summit node.

We identify the assumptions by running strong scaling numbers of a
single large 3D-FFT (default is 1024^3) distributed with the
slab-pencil (1D) decomposition. From a communication point of view,
bandwidth is expected to be the bottleneck when the number of nodes is
small. However, latency is expected to be the bottleneck when the
number of nodes is large. Timing breakdown of the individual
components of the FFT is also collected to identify the impact of
packing and computation on the overall computation time.

### Compilation
Before building, update `CUDA_PATH` in the `Makefile` to the CUDA install path you'd like to use.
``` 
make                    #creates executable distributed-3dfft.x
```

### Execution (on Summit)
A job script is included for running on Summit. **You will need to update the script with a valid project number.** 

To run an experiment with multiple different sizes on Summit, replace the job with the following commands:
```
for resources in 2 4 8 16 32 64 128 256; do
  jsrun --smpiargs="-gpu" --nrs $resources --tasks_per_rs 1 --cpu_per_rs 6 --gpu_per_rs 6 --rs_per_host 1 ./distributed-3dfft.x
done
```

### Outputs
The output of each of the experiment is in Common Separated Value (CSV) format. The fields reported
are as follows:
```
Rank, gpu_id, start_time,  stg1_end, stg1_gpu_to_host, start_pack, pack_end,
 a2a_end,  copy_host_gpu,   end_time,   total_time
```
Execution time for the following is computed as follows:
1. Stage 1 (2D FFT) execution : stg1_end - start_time
2. Copying from GPU to host   : stg1_gpu_to_host - stg1_end
3. Packing                    : pack_end - start_pack
4. Communication              : a2a_end - pack_end
5. Copying from host to GPU   : copy_host_gpu - a2a_end
6. Stage 2 (1D FFT) execution : end_time - copy_host_gpu
