/**
 * Global Memory Bitoic Sort
 *
 * This uses gpu global memory to sort arrays to sort long arrays of ints
 *
 * Author: Andrew Boessen
 */

#include "bitonic_sort.cuh"

/**
 * Swap
 *
 * This is used for swapping elements in bitonic sorting
 *
 * @param x caller line id's value
 * @param mask source lane id = caller line id ^ mask
 * @param dir direction to swap
 * @param arr global memory array
 *
 */
__device__ void swap(int x, int mask, int dir, int *arr) {
  // get correspondin element to x in butterfly diagram
  int y = x ^ mask;
  // lower ids thread perform swap
  if (y > x) {
    if (dir) {
      // sort ascending
      if (arr[x] < arr[y]) {
        int temp = arr[x];
        arr[x] = arr[y];
        arr[y] = temp;
      }
    } else {
      // sort descending
      if (arr[x] > arr[y]) {
        int temp = arr[x];
        arr[x] = arr[y];
        arr[y] = temp;
      }
    }
  }
}

/**
 * Global Memory Bitonic Sort
 *
 * The function uses the butterfly network pattern of bitonic sort, leveraging
 * CUDA's warp-level primitives for efficient sorting within a warp (32
 * threads). The swaps are tiled into warps of 32 threads. This is able to do
 * swaps without allocating extra memory for temporary variable.
 *
 * @param arr Pointer to the array of integers to be sorted
 * @param size Total number of elements in the array
 *
 * @note This function assumes that the number of threads per block is at least
 * equal to the warp size. Elements beyond the array size are padded with
 * INT_MAX.
 *
 * @see swap() for the element comparison and swapping logic
 */
__global__ void globalBitonicSort(int *arr, int size) {
  // local thread id in block
  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

  // make bitonic sequence and sort
  for (int i = 0; (1 << i) <= blockDim.x; i++) {
    for (int j = 1; j <= i; j++) {
      // distance between caller and source lanes
      int mask = 1 << (i - j);

      // perform compare and swap
      int dir = thread_id & (1 << i);
      swap(thread_id, mask, dir, arr);
      __syncthreads();
    }
  }
}

void launchBitonicSort(int *arr, int size) {
  const int BLOCK_SIZE = 512;
  const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  globalBitonicSort<<<NUM_BLOCKS, BLOCK_SIZE>>>(arr, size);
}
