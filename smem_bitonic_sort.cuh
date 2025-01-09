#ifndef SMEM_BITONIC_SORT_CUH
#define SMEM_BITONIC_SORT_CUH

#include <climits>
#include <cuda_runtime.h>

__device__ int swap(int x, int mask, int dir);
__global__ void smemBitonicSort(int *arr, int size);
void launchSmemBitonicSort(int *arr, int size);

#endif // SMEM_BITONIC_SORT_CUH
