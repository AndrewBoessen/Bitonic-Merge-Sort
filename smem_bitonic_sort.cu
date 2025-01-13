/**
 * SMEM Bitoic Sort
 *
 * This uses shared memory to sort arrays. This uses warp shffle operator to
 * compare and swap
 *
 * Author: Andrew Boessen
 */

#include "smem_bitonic_sort.cuh"

/**
 * Swap
 *
 * This is used for swapping elements in bitonic sorting
 *
 * @param x caller line id's value
 * @param mask source lane id = caller line id ^ mask
 * @param dir direction to swap
 *
 * @return min or max of source and caller
 */
__device__ int swap(int x, int mask, int dir) {
  // get correspondin element to x in butterfly diagram
  int y = __shfl_xor_sync(0xffffffff, x, mask);
  // return smaller or larger value based on direction of swap
  return x < y == dir ? y : x;
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
  __shared__ int smem[1 << 10];

  // local thread id in block
  int thread_id = threadIdx.x;

  // seed shared memory array with value from global array
  // pad overflow threads with INT_MAX
  smem[thread_id] = thread_id < size ? arr[thread_id] : INT_MAX;

  // value in array
  int x;

  // make bitonic sequence and sort
  for (int i = 0; (1 << i) <= blockDim.x; i++) {
    for (int j = 0; j <= i; j++) {
      // distance between caller and source lanes
      int offset = 1 << (i - j);
      // id into smem array
      int arr_id = (thread_id % 2 * offset / 2) +
                   (thread_id / offset * offset) + (thread_id % 4 / 2);
      // direction to swap caller and source lanes
      int dir;

      // mask for swap offset
      int mask;

      // only alternate direction when forming bitonic sequence
      if (1 << i == blockDim.x) {
        dir = (arr_id >> (i - j)) & 1;
      } else {
        dir = (arr_id >> (i + 1)) & 1 ^ (arr_id >> (i - j)) & 1;
      }
      // use registers for smaller than warp size
      // otherwize load from smem
      if (1 << j == warpSize) {
        x = smem[thread_id];
      } else if (1 << j > warpSize) {
        x = smem[arr_id];
      }

      if (1 << j <= warpSize) {
        mask = 1 << (i - j);
      } else {
        // elements to compare and swap are directly next to eachother in warp
        mask = 1;
      }
      // perform compare and swap
      x = swap(x, mask, dir);
      // store in smem
      if (1 << j > warpSize) {
        smem[arr_id] = x;
        // wait for all warps to finish swap before going to next layer
        __syncthreads();
      }
    }
  }

  // update value in array with sorted value
  if (thread_id < size) {
    arr[thread_id] = x;
  }
}

void launchSmemBitonicSort(int *arr, int size) {
  const int BLOCK_SIZE = 1024;
  smemBitonicSort<<<size / BLOCK_SIZE, BLOCK_SIZE>>>(arr, size);
}
