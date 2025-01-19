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
  extern __shared__ int smem[];

  // local thread id in block
  int thread_id = threadIdx.x;

  // seed shared memory array with value from global array
  // pad overflow threads with INT_MAX
  smem[thread_id] = thread_id < size ? arr[thread_id] : INT_MAX;
  __syncthreads();

  // make bitonic sequence and sort
  for (int i = 0; (1 << i) <= size; i++) {
    for (int j = 0; j <= i; j++) {
      // distance between caller and source lanes
      int offset = 1 << (i - j - 1);
      // direction to swap caller and source lanes
      int dir;
      // only alternate direction when forming bitonic sequence
      if (1 << i == blockDim.x) {
        dir = (thread_id >> (i - j)) & 1;
      } else {
        dir = (thread_id >> (i + 1)) & 1 ^ (thread_id >> (i - j)) & 1;
      }
      if (1 << i <= warpSize) {
        smem[thread_id] = swap(smem[thread_id], offset, dir);
      } else {
        __syncthreads();
        int partner_val = smem[thread_id ^ offset];
        int val = smem[thread_id];
        // compare and swap elements
        smem[thread_id] = val < partner_val == dir ? val : partner_val;
        smem[thread_id ^ offset] = val < partner_val == dir ? partner_val : val;
      }
    }
  }
  __syncthreads();

  // update value in array with sorted value
  if (thread_id < size) {
    arr[thread_id] = smem[thread_id];
  }
}

void launchBitonicSort(int *arr, int size) {
  const int BLOCK_SIZE = 1024;
  smemBitonicSort<<<size / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(
      arr, size);
}
