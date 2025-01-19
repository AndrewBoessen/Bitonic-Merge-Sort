#ifndef BITONIC_SORT_CUH
#define BITONIC_SORT_CUH

#include <climits>
#include <cuda_runtime.h>

__device__ int swap(int x, int mask, int dir);
__global__ void warpBitonicSort(int *arr, int size);
__global__ void smemBitonicSort(int *arr, int size);
void launchBitonicSort(int *arr, int size);

#endif // BITONIC_SORT_CUH
