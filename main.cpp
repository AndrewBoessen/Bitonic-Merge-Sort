#include <stdio.h>
#include <stdlib.h>
#include "warp_bitonic_sort.cuh"

// Function to check if the array is sorted
bool isSorted(int* arr, int size) {
    for (int i = 1; i < size; i++) {
        if (arr[i] < arr[i-1]) return false;
    }
    return true;
}

int main() {
    const int SIZE = 1024; // Must be a multiple of 32 for this example
    const int BLOCK_SIZE = 256;

    // Allocate and initialize host array
    int* h_arr = new int[SIZE];
    for (int i = 0; i < SIZE; i++) {
        h_arr[i] = rand() % 1000; // Random integers between 0 and 999
    }

    // Allocate device array
    int* d_arr;
    cudaMalloc(&d_arr, SIZE * sizeof(int));

    // Copy host array to device
    cudaMemcpy(d_arr, h_arr, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    launchWarpBitonicSort(d_arr, SIZE);

    // Copy result back to host
    cudaMemcpy(h_arr, d_arr, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // Check if sorted
    bool sorted = isSorted(h_arr, SIZE);
    printf("Array is %s\n", sorted ? "sorted" : "not sorted");

    // Print first few elements to verify
    printf("First 10 elements: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Clean up
    delete[] h_arr;
    cudaFree(d_arr);

    return 0;
}
