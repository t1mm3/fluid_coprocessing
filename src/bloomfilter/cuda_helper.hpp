#pragma once

#include <cub/cub.cuh>
#include <cuda/api_wrappers.h>
#include <cuda_runtime.h>
#include <dtl/dtl.hpp>

#if !defined(__global__)
#define __global__
#endif

static constexpr u32 warp_size = 32;

// get temporary memory allocation to perform the radix sort
template <typename Key, typename Value> size_t get_temp_storage_requirement(size_t data_length) {
	size_t temp_storage_requirement{0};
	auto status = cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_requirement, static_cast<Key *>(nullptr),
	                                              static_cast<Key *>(nullptr), static_cast<Value *>(nullptr),
	                                              static_cast<Value *>(nullptr), data_length, 24, 32);
	cuda::throw_if_error(status, "CUB could not determine the temporary storage "
	                             "size requirement for sorting key-value pairs");
	return temp_storage_requirement;
}

//===----------------------------------------------------------------------===//
template <typename Key> size_t get_temp_storage_requirement(size_t data_length) {
	size_t temp_storage_requirement{0};
	auto status = cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_requirement, static_cast<Key *>(nullptr),
	                                             static_cast<Key *>(nullptr), data_length, 24, 32);
	if (status != cudaSuccess)
		std::cerr << "CUB could not determine the temporary storage size "
		             "requirement for sorting key-value pairs"
		          << std::endl;
	return temp_storage_requirement;
}
//===----------------------------------------------------------------------===//

template <typename Key, typename Value> void sort_with_cub(size_t data_length, Key host_keys, Value host_values) {
	if (data_length > std::numeric_limits<int>::max()) {
		throw std::logic_error("CUB only accepts lengths which fit in an int-type variable");
	}

	auto temp_storage_size = get_temp_storage_requirement<Key, Value>(data_length);

	auto current_device = cuda::device::current::get();

	auto device_keys = cuda::memory::device::make_unique<Key[]>(current_device, data_length);
	auto device_values = cuda::memory::device::make_unique<Value[]>(current_device, data_length);
	auto sorted_device_keys = cuda::memory::device::make_unique<Key[]>(current_device, data_length);
	auto sorted_device_values = cuda::memory::device::make_unique<Value[]>(current_device, data_length);
	auto temp_storage = cuda::memory::device::make_unique<unsigned char[]>(current_device, temp_storage_size);
	// Note: in C++20, we would have used std::byte

	auto keys_size = data_length * sizeof(Key);
	auto values_size = data_length * sizeof(Key);

	cuda::memory::copy(&(*std::begin(host_keys)), device_keys.get(), keys_size);
	cuda::memory::copy(&(*std::begin(host_values)), device_values.get(), values_size);

	auto status = cub::DeviceRadixSort::SortPairs(temp_storage.get(), temp_storage_size, device_keys.get(),
	                                              sorted_device_keys.get(), device_values.get(),
	                                              sorted_device_values.get(), data_length);
	cuda::throw_if_error(status, "CUB Radix key-value-pair sorting failed");
}

/// prints GPU information
void get_device_properties() {
	$i32 device_cnt = 0;
	cudaGetDeviceCount(&device_cnt);
	cudaDeviceProp device_prop;

	for (int i = 0; i < device_cnt; i++) {
		cudaGetDeviceProperties(&device_prop, i);
		std::cout << "\n+----------------------------------------------------------"
		             "---------------------+\n";
		printf("|  Device id: %d\t", i);
		printf("  Device name: %s\t", device_prop.name);
		printf("  Compute capability: %d.%d\n", device_prop.major, device_prop.minor);
		std::cout << std::endl;
		printf("|  Memory Clock Rate [KHz]: %d\n", device_prop.memoryClockRate);
		printf("|  Memory Bus Width [bits]: %d\n", device_prop.memoryBusWidth);
		printf("|  Peak Memory Bandwidth [GB/s]: %f\n",
		       2.0 * device_prop.memoryClockRate * (device_prop.memoryBusWidth / 8) / 1.0e6);
		printf("|  L2 size [KB]: %d\n", device_prop.l2CacheSize / 1024);
		std::cout << std::endl;
		printf("|  Number of SMs: %d\n", device_prop.multiProcessorCount);
		printf("|  Max. number of threads per SM: %d\n", device_prop.maxThreadsPerMultiProcessor);
		printf("|  Concurrent kernels: %d\n", device_prop.concurrentKernels);
		printf("|  warpSize: %d\n", device_prop.warpSize);
		printf("|  maxThreadsPerBlock: %d\n", device_prop.maxThreadsPerBlock);
		printf("|  maxThreadsDim[0]: %d\n", device_prop.maxThreadsDim[0]);
		printf("|  maxGridSize[0]: %d\n", device_prop.maxGridSize[0]);
		printf("|  pageableMemoryAccess: %d\n", device_prop.pageableMemoryAccess);
		printf("|  concurrentManagedAccess: %d\n", device_prop.concurrentManagedAccess);
		printf("|  Number of async. engines: %d\n", device_prop.asyncEngineCount);
		std::cout << "\n+----------------------------------------------------------"
		             "--------------------+\n";
	}
}

/// returns the global id of the executing thread
__device__ __forceinline__ u32 global_thread_id() {
	return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ u32 global_size() {
	return gridDim.x * blockDim.x;
}

__device__ __forceinline__ u32 warp_cnt() {
	return (global_size() + (warp_size - 1)) / warp_size;
}

__device__ __forceinline__ u32 block_size() {
	return blockDim.x;
}

/// returns the block id in [0, grid_size)
__device__ __forceinline__ u32 block_id() {
	return blockIdx.x;
}

/// returns the thread id within the current block: [0, block_size)
__device__ __forceinline__ u32 block_local_thread_id() {
	return threadIdx.x;
}

/// returns the warp id within the current block
/// the id is in [0, u), where u = block_size / warp_size
__device__ __forceinline__ u32 block_local_warp_id() {
	return block_local_thread_id() / warp_size;
}

/// returns the warp id (within the entire grid)
__device__ __forceinline__ u32 global_warp_id() {
	return global_thread_id() / warp_size;
}

/// returns the thread id [0,32) within the current warp
__device__ __forceinline__ u32 warp_local_thread_id() {
	return block_local_thread_id() % warp_size;
}

// taken from:
// https://codeyarns.com/2011/03/02/how-to-do-error-checking-in-cuda/
// __forceinline__ void __cuda_check_error(const char *file, const int line) {
// 	cudaError err = cudaGetLastError();
// 	if (cudaSuccess != err) {
// 		fprintf(stderr, "cuda_check_error() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
// 		exit(-1);
// 	}

// 	//  // More careful checking. However, this will affect performance.
// 	//  // Comment away if needed.
// 	//  err = cudaDeviceSynchronize();
// 	//  if (cudaSuccess != err) {
// 	//      fprintf(stderr, "cuda_check_error() with sync failed at %s:%i : %s\n",
// 	//               file, line, cudaGetErrorString(err) );
// 	//      exit(-1);
// 	//  }
// 	//  return;
// }
