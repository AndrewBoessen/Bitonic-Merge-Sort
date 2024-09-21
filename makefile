CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++11 -O2
NVCCFLAGS = -O2
CUDA_PATH = /opt/cuda
INCLUDES = -I$(CUDA_PATH)/include
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart

all: warp_bitonic_sort

warp_bitonic_sort: main.o warp_bitonic_sort.o
	$(CXX) $^ -o $@ $(LDFLAGS)

main.o: main.cpp warp_bitonic_sort.cuh
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

warp_bitonic_sort.o: warp_bitonic_sort.cu warp_bitonic_sort.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f *.o warp_bitonic_sort
