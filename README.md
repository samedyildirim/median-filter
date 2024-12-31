# High Performance Median Filtering on CUDA

This repository includes two median filtering algorithms implemented on CUDA. The both algorithms are presented in my master thesis with title of *A Parallel Median Filtering Algorithm for SIMD Architecture*.

The first algorithm presented in Chapter 2 with the title of *Accelerated Median Filtering via Column-Wise Parallel Histogram Update* in the thesis is in the file `src/median_filter-algorithm1.cu`.

The first algorithm was presented at *17th International Conference on Machine Vision (ICMV 2024)* in Edinburg, the UK. The title of the paper is *High-Throughput Median Filtering for Large Kernel Sizes on CUDA*.

The second algorithm presented in Chapter 3 with the title of *Advanced Parallelism Techniques Using Multi-Histogram Multi-Segment Processing in Median Filtering* in the thesis is in the file `median_filter-algorithm2.cu`.

The median filtering algorithms were tested on an NVIDIA H100 Tensor Core GPU with 80 GB memory. The second algorithm in the thesis outperformed Green's implementation and OpenCV-CPU for all kernel sizes, ArrayFire starting from filter size of 5, NPP starting from 7, and PRMF starting from 9. The algorithm successfully maintained its performance with increasing kernel sizes, and starting from the filter size of 9, it was the fastest among all algorithms. And it provided a speedup of up to 534× and 180× relative to OpenCV-CPU and Green, respectively.

The paper and the thesis with experimental results will be added to the repository soon.
