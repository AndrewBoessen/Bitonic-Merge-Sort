# GPU Bitonic Merge Sort

This repository contains an implementation of the Bitonic Merge Sort algorithm optimized for GPU execution. Bitonic Merge Sort is a parallel sorting algorithm that is particularly well-suited for implementation on GPUs due to its highly parallelizable nature.

## Overview

Bitonic Merge Sort is a comparison-based sorting algorithm that can sort n elements in O(log^2(n)) parallel steps using O(n log^2(n)) comparisons. This implementation leverages the massive parallelism of GPUs to achieve high-performance sorting for large datasets.

## Features

- Efficient GPU implementation of Bitonic Merge Sort
- Support for sorting large arrays of various data types
- Optimized for CUDA-capable NVIDIA GPUs
- Configurable block size and thread count for performance tuning
- Benchmarking tools to compare CPU and GPU sorting performance

## Requirements

- CUDA-capable GPU (Compute Capability 3.0 or higher)
- CUDA Toolkit (version 10.0 or later recommended)
- C++ compiler with C++11 support
- CMake (version 3.10 or later)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/gpu-bitonic-sort.git
   cd gpu-bitonic-sort
   ```

2. Create a build directory and run CMake:
   ```
   mkdir build && cd build
   cmake ..
   ```

3. Build the project:
   ```
   make
   ```

## Usage

After building the project, you can run the main executable:

```
./gpu_bitonic_sort [array_size]
```

This will generate a random array of the specified size, sort it using both CPU and GPU implementations, and display timing information and correctness verification.

## Performance

The GPU implementation typically outperforms CPU-based sorting algorithms for large datasets. Performance can vary based on the specific GPU hardware and the size of the input data. Benchmark results for various array sizes and GPU models can be found in the `benchmarks` directory.

## Contributing

Contributions to improve the implementation or extend its functionality are welcome. Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

