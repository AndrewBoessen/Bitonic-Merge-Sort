/**
 * Global Memory Bitoic Sort
 *
 * This uses gpu global memory to sort arrays to sort long arrays of ints
 *
 * Author: Andrew Boessen
 */

#include "bitonic_sort.cuh"

/**
 * Global Memory Bitonic Sort Swap
 *
 * This is used for swapping elements in bitonic sorting
 *
 * @param x caller line id's value
 * @param i current large step in bitonic sort sequence
 * @param j current small step in sequence
 * @param arr global memory array
 *
 */
__global__ void globalSwap(int i, int j, int *arr) {
  // thread id within grid
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  // distance between caller and source lanes
  int mask = 1 << (i - j);

  // perform compare and swap
  int dir = x & (1 << i);

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
 * @param arr Pointer to the array of integers to be sorted
 * @param size Total number of elements in the array
 * @param block_size Number of threads in one block
 * @param num_blocks Number of total block in grid
 *
 * @note This function assumes that the number elements in the arrays is a power
 * of two
 *
 * @see globalSwap() for the element comparison and swapping logic kernel
 */
void globalBitonicSort(int *arr, int size, int block_size,
                       int num_blocks) { // make bitonic sequence and sort
  for (int i = 0; (1 << i) <= size; i++) {
    for (int j = 1; j <= i; j++) {
      globalSwap<<<num_blocks, block_size>>>(i, j, arr);
    }
  }
}
void launchBitonicSort(int *arr, int size) {
  const int BLOCK_SIZE = 512;
  const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  // call sort function
  globalBitonicSort(arr, size, BLOCK_SIZE, NUM_BLOCKS);
}
