#ifndef WARP_BITONIC_SORT_CUH
#define WARP_BITONIC_SORT_CUH

#include <cuda_runtime.h>
#include <climits>

__device__ int swap(int x, int mask, int dir);
__global__ void warpBitonicSort(int *arr, int size);
void launchWarpBitonicSort(int *arr, int size);

#endif // WARP_BITONIC_SORT_CUH
