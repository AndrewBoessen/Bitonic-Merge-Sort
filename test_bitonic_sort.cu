#include "warp_bitonic_sort.cuh"
#include <algorithm>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

// Function to check if the array is sorted
bool isSorted(int *arr, int size) {
  for (int i = 1; i < size; i++) {
    if (arr[i] < arr[i - 1])
      return false;
  }
  return true;
}

// Test fixture for CUDA Bitonic Sort
class BitonicSortTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize CUDA
    cudaSetDevice(0);
  }

  void TearDown() override {
    // Clean up CUDA
    cudaDeviceReset();
  }

  // Helper function to generate random arrays
  std::vector<int> generateRandomArray(int size) {
    std::vector<int> arr(size);
    for (int i = 0; i < size; i++) {
      arr[i] = rand() % 1000; // Random integers between 0 and 999
    }
    return arr;
  }

  // Helper function to compare two arrays
  bool compareArrays(int *arr1, int *arr2, int size) {
    for (int i = 0; i < size; i++) {
      if (arr1[i] != arr2[i])
        return false;
    }
    return true;
  }
};

// Test case for sorting a small array
TEST_F(BitonicSortTest, SmallArraySort) {
  const int SIZE = 32; // Must be a multiple of 32 for warp-level sorting
  std::vector<int> h_arr = generateRandomArray(SIZE);
  std::vector<int> h_arr_sorted = h_arr;
  std::sort(h_arr_sorted.begin(), h_arr_sorted.end());

  int *d_arr;
  cudaMalloc(&d_arr, SIZE * sizeof(int));
  cudaMemcpy(d_arr, h_arr.data(), SIZE * sizeof(int), cudaMemcpyHostToDevice);

  launchWarpBitonicSort(d_arr, SIZE);

  cudaMemcpy(h_arr.data(), d_arr, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

  EXPECT_TRUE(isSorted(h_arr.data(), SIZE));
  EXPECT_TRUE(compareArrays(h_arr.data(), h_arr_sorted.data(), SIZE));

  cudaFree(d_arr);
}

// Test case for sorting a large array
TEST_F(BitonicSortTest, LargeArraySort) {
  const int SIZE = 4096; // Must be a multiple of 32 for warp-level sorting
  std::vector<int> h_arr = generateRandomArray(SIZE);
  std::vector<int> h_arr_sorted = h_arr;
  std::sort(h_arr_sorted.begin(), h_arr_sorted.end());

  int *d_arr;
  cudaMalloc(&d_arr, SIZE * sizeof(int));
  cudaMemcpy(d_arr, h_arr.data(), SIZE * sizeof(int), cudaMemcpyHostToDevice);

  launchWarpBitonicSort(d_arr, SIZE);

  cudaMemcpy(h_arr.data(), d_arr, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

  EXPECT_TRUE(isSorted(h_arr.data(), SIZE));
  EXPECT_TRUE(compareArrays(h_arr.data(), h_arr_sorted.data(), SIZE));

  cudaFree(d_arr);
}

// Test case for sorting an already sorted array
TEST_F(BitonicSortTest, AlreadySortedArray) {
  const int SIZE = 256; // Must be a multiple of 32 for warp-level sorting
  std::vector<int> h_arr(SIZE);
  for (int i = 0; i < SIZE; i++) {
    h_arr[i] = i;
  }

  int *d_arr;
  cudaMalloc(&d_arr, SIZE * sizeof(int));
  cudaMemcpy(d_arr, h_arr.data(), SIZE * sizeof(int), cudaMemcpyHostToDevice);

  launchWarpBitonicSort(d_arr, SIZE);

  cudaMemcpy(h_arr.data(), d_arr, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

  EXPECT_TRUE(isSorted(h_arr.data(), SIZE));

  cudaFree(d_arr);
}

// Test case for sorting a reverse-sorted array
TEST_F(BitonicSortTest, ReverseSortedArray) {
  const int SIZE = 512; // Must be a multiple of 32 for warp-level sorting
  std::vector<int> h_arr(SIZE);
  for (int i = 0; i < SIZE; i++) {
    h_arr[i] = SIZE - i;
  }

  int *d_arr;
  cudaMalloc(&d_arr, SIZE * sizeof(int));
  cudaMemcpy(d_arr, h_arr.data(), SIZE * sizeof(int), cudaMemcpyHostToDevice);

  launchWarpBitonicSort(d_arr, SIZE);

  cudaMemcpy(h_arr.data(), d_arr, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

  EXPECT_TRUE(isSorted(h_arr.data(), SIZE));

  cudaFree(d_arr);
}

// Test case for sorting an array with duplicate elements
TEST_F(BitonicSortTest, ArrayWithDuplicates) {
  const int SIZE = 1024; // Must be a multiple of 32 for warp-level sorting
  std::vector<int> h_arr = generateRandomArray(SIZE);
  for (int i = 0; i < SIZE / 2; i++) {
    h_arr[i] = h_arr[i + SIZE / 2]; // Introduce duplicates
  }

  std::vector<int> h_arr_sorted = h_arr;
  std::sort(h_arr_sorted.begin(), h_arr_sorted.end());

  int *d_arr;
  cudaMalloc(&d_arr, SIZE * sizeof(int));
  cudaMemcpy(d_arr, h_arr.data(), SIZE * sizeof(int), cudaMemcpyHostToDevice);

  launchWarpBitonicSort(d_arr, SIZE);

  cudaMemcpy(h_arr.data(), d_arr, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

  EXPECT_TRUE(isSorted(h_arr.data(), SIZE));
  EXPECT_TRUE(compareArrays(h_arr.data(), h_arr_sorted.data(), SIZE));

  cudaFree(d_arr);
}

// Test case for sorting an array with a single element
TEST_F(BitonicSortTest, SingleElementArray) {
  const int SIZE = 1;
  std::vector<int> h_arr = {42};

  int *d_arr;
  cudaMalloc(&d_arr, SIZE * sizeof(int));
  cudaMemcpy(d_arr, h_arr.data(), SIZE * sizeof(int), cudaMemcpyHostToDevice);

  launchWarpBitonicSort(d_arr, SIZE);

  cudaMemcpy(h_arr.data(), d_arr, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

  EXPECT_TRUE(isSorted(h_arr.data(), SIZE));

  cudaFree(d_arr);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
