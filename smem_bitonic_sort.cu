/**
 * SMEM Bitoic Sort
 *
 * This uses shared memory to sort arrays. This uses warp shffle operator to
 * compare and swap
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
 * @param arr shared memory
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
 * SMEM Bitonic Sort
 *
 * This function performs a bitonic sort on integers whithin a thread blocks of
 * 1024 threads. This stores itermediate products in shared memory for better
 * efficiency.
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
__global__ void smemBitonicSort(int *arr, int size) {
  // shared memory for block of 1024 threads
  extern __shared__ int smem[];

  // local thread id in block
  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  // id if thread within its block
  int local_id = threadIdx.x;

  // seed shared memory array with value from global array
  // pad overflow threads with INT_MAX
  smem[local_id] = thread_id < size ? arr[thread_id] : INT_MAX;
  __syncthreads();

  // make bitonic sequence and sort
  for (int i = 0; (1 << i) <= blockDim.x; i++) {
    for (int j = 1; j <= i; j++) {
      // distance between caller and source lanes
      int mask = 1 << (i - j);

      // perform compare and swap
      int dir = local_id & (1 << i);
      swap(local_id, mask, dir, smem);
      __syncthreads();
    }
  }

  // update value in array with sorted value
  if (thread_id < size) {
    arr[thread_id] = smem[local_id];
  }
  __syncthreads();
}

void launchBitonicSort(int *arr, int size) {
  const int BLOCK_SIZE = 512;
  smemBitonicSort<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE,
                    BLOCK_SIZE * sizeof(int)>>>(arr, size);
}
