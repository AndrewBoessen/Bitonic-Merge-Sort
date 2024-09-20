/**
* Warp Bitoic Sort
*
* This uses warp shuffle to sort integers in a warp with bitonic sort
*
* Author: Andrew Boessen
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <climits>

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
  int y = __shfl_xor_sync(x, mask);
  // return smaller or larger value based on direction of swap
  return x < y == dir ? y : x;
}

/**
 * Warp Bitonic Sort
 *
 * This function performs a bitonic sort on integers within a warp using warp shuffle operations.
 * It sorts a portion of the input array corresponding to the calling thread's warp.
 *
 * The function uses the butterfly network pattern of bitonic sort, leveraging CUDA's warp-level
 * primitives for efficient sorting within a warp (32 threads).
 *
 * @param arr Pointer to the array of integers to be sorted
 * @param size Total number of elements in the array
 *
 * @note This function assumes that the number of threads per block is at least equal to the warp size.
 *       Elements beyond the array size are padded with INT_MAX.
 *
 * @see swap() for the element comparison and swapping logic
 */
__global__ void warpBitonicSort(int *arr, int size) {
  // threadIdx.x % 32
  int lane_id = threadIdx.x & 0x1f;
  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  // seed x with value from array
  int x = thread_id < size ? arr[thread_id] : INT_MAX;

  // make bitonic sequence and sort
  for (int i = 0; (1 << i) <= warpSize; i++) {
    for (int j = 0; j <= i; j++) {
      // distance between caller and source lanes
      int mask = 1 << (i-j);
      // direction to swap caller and source lanes
      int dir;
      // only alternate direction when forming bitonic sequence
      if (1 << i == warpSize) {
        dir = (lane_id >> (i - j)) & 1;
      } else {
        dir = (lane_id >> (i + 1)) & 1 ^ (lane_id >> (i - j)) & 1;
      }
      x = swap(x, mask, dir);
    }
  }

  // update value in array with sorted value
  if (thread_id < size) {
    arr[thread_id] = x;
  }
}
