#include <algorithm>
#include <iostream>
#include <iterator>
#include "../cuda_helper.hpp"
#include "kernels.hpp"
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#define BEGIN_BIT 24
#define END_BIT 32

//===----------------------------------------------------------------------===//

template <typename T>
struct is_true : public thrust::unary_function<T,bool>
{
    __host__ __device__
    bool operator()(T x)
    {
        return x == 1;
    }
};

template <typename filter_t>
struct cuda_filter {
  struct perf_data_t {
	$i32 cuda_block_size = 0;
	$i32 cuda_block_cnt = 0;
	$u64 sort_throughput = 0;
	double sort_time = 0;
	$u64 hash_throughput = 0;
	double hash_time = 0;
	$u64 probes_per_second = 0;
	double probe_time = 0;
	$u64 total_throughput = 0;
	double candidate_time = 0;
  };

  using key_t = typename filter_t::key_t;
  using word_t = typename filter_t::word_t;
  /// instance of the host-side filter
  const filter_t &filter;
  /// pointer to the filter data on the device
  word_t *device_word_array;
  /// size of the filter data in bytes
  $u64 word_array_size;

  /// c'tor
  cuda_filter(const filter_t &filter, const word_t *__restrict word_array,
			  const std::size_t word_cnt)
	  : filter(filter) {
	cudaSetDevice(0);
	// copy the filter data to device memory
    assert(word_cnt > 0);
	word_array_size = word_cnt * sizeof(word_t);
	cudaMalloc((void **)&device_word_array, word_array_size);
	cuda_check_error();
	cudaMemcpy(device_word_array, word_array, word_array_size,
			   cudaMemcpyHostToDevice);
	cuda_check_error();
  }

  /// d'tor
  ~cuda_filter() {
	cudaFree(device_word_array);
	cuda_check_error();
  }
  /// batch-probe the filter
  void contains(u32 *__restrict__ keys, u32 key_cnt, $u32 *__restrict__ bitmap,
				$u32 *__restrict__ /*keys_hash*/) {
	perf_data_t perf_data;
	contains_naive(keys, key_cnt, bitmap, perf_data);
  }

  /// batch-probe the filter with profile
  void contains_clustering(u32 *__restrict__ keys, u32 key_cnt, $u32 *__restrict__ bitmap,
				perf_data_t &perf_data, std::size_t bits_to_sort ) {
	// Allocate device memory and copy the keys to device
	$u32 *d_keys, *d_keys_hash, *d_positions;
	u32 device_keys_size = sizeof($u32) * key_cnt;
	const std::size_t gran = 4ull * 1024;
	const std::size_t device_keys_alloc_size = ((device_keys_size + (gran - 1)) / gran) * gran;


	// Allocate device memory for hash_values and positions
    cudaMalloc((void **)&d_keys,      device_keys_alloc_size);
	cudaMalloc((void **)&d_keys_hash, device_keys_alloc_size);
	cudaMalloc((void **)&d_positions, device_keys_alloc_size);
	cuda_check_error();

	u64 repeats = 10;
	cudaMemcpy(d_keys, keys, device_keys_size, cudaMemcpyHostToDevice);

	// Calculate Hash Values

	i32 block_size = 32;
	i32 block_cnt = (key_cnt + block_size - 1) / block_size;
	auto start_hash = std::chrono::high_resolution_clock::now();

    calculate_hash_kernel<<<block_cnt, block_size>>>(filter, d_keys, key_cnt, d_keys_hash, d_positions);
    cudaDeviceSynchronize();

	auto end_hash = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration_hash = end_hash - start_hash;
	perf_data.hash_time = duration_hash.count();
	perf_data.hash_throughput = static_cast<u64>(key_cnt / duration_hash.count());
	cudaFree(d_keys);

	// Sort Hash Values by 8 MSB

	$u32 *d_sorted_keys, *d_sorted_positions;
	cudaMalloc((void **)&d_sorted_keys, device_keys_alloc_size);
	cudaMalloc((void **)&d_sorted_positions, device_keys_alloc_size);
	cuda_check_error();

    // Sorting
	auto temp_storage_size = get_temp_storage_requirement<key_t, key_t>(key_cnt);

	unsigned char *temp_storage;
	cudaMalloc((void **)&temp_storage, temp_storage_size);

	auto start_sort = std::chrono::high_resolution_clock::now();

	auto status = cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_size, d_keys_hash, d_sorted_keys, d_positions, d_sorted_positions, key_cnt, (END_BIT - bits_to_sort), END_BIT);
	cudaDeviceSynchronize();

	auto end_sort = std::chrono::high_resolution_clock::now();
	cudaFree(d_keys_hash);
	cudaFree(d_positions);
	cudaFree(temp_storage);

	std::chrono::duration<double> duration_sort = end_sort - start_sort;
	perf_data.sort_time = duration_sort.count();
	perf_data.sort_throughput = static_cast<u64>(key_cnt / duration_sort.count());

	// Allocate memory for the result bitmap
	$u32 *device_bitmap;
    const std::size_t device_bitmap_size = (key_cnt) * sizeof($u32); // one position per key
    cudaMalloc((void **)&device_bitmap, device_bitmap_size);
	cuda_check_error();

	// Probe with sorted hashes
	i32 elements_per_thread = warp_size;
	i32 elements_per_block = block_size * elements_per_thread;
	i32 block_count = (key_cnt + elements_per_block - 1) / elements_per_block;
	perf_data.cuda_block_size = block_size;
	perf_data.cuda_block_cnt = block_count;

	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	//warm up
	if(repeats > 1) {
        contains_clustered_kernel<<<block_count, block_size>>>(
            filter, device_word_array, d_keys, d_sorted_keys, key_cnt, device_bitmap,
            d_sorted_positions);
    }
	// Real Experiment
	auto start_probe = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < repeats; i++) {
        contains_clustered_kernel<<<block_count, block_size>>>(
            filter, device_word_array, d_keys, d_sorted_keys, key_cnt, device_bitmap,
            d_sorted_positions);
        cudaDeviceSynchronize();
		cuda_check_error();
    }
	

	auto end_probe = std::chrono::high_resolution_clock::now();
	perf_data.probe_time = std::chrono::duration<double>(end_probe - start_probe).count() / repeats;
	u64 probes_per_second = static_cast<u64>((key_cnt) / perf_data.probe_time);
	perf_data.probes_per_second = probes_per_second;

	uint32_t *result_bitmap;
	cudaMalloc((void**)&result_bitmap, device_bitmap_size);

	auto start_candidates = std::chrono::high_resolution_clock::now();
	auto output_end = thrust::copy_if(thrust::device, d_sorted_positions, d_sorted_positions + key_cnt, device_bitmap, result_bitmap, is_true<uint32_t>());
	auto end_candidates = std::chrono::high_resolution_clock::now();
	perf_data.candidate_time = std::chrono::duration<double>(end_candidates - start_candidates).count();

	double total_time = static_cast<double>(perf_data.hash_time + perf_data.sort_time + perf_data.probe_time + perf_data.candidate_time);
	perf_data.total_throughput = static_cast<u64>((key_cnt) / total_time);
	
	//copy back only the candidate list
	cudaMemcpy(bitmap, result_bitmap, (output_end - result_bitmap)  * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cuda_check_error();
	cudaDeviceSynchronize();
	cuda_check_error();

	// Free temporary resources
	cudaDeviceSynchronize();
	cudaFree(d_sorted_keys);
	cudaFree(d_sorted_positions);
	cudaFree(device_bitmap);
	cudaFree(result_bitmap);
	cuda_check_error();
  }

  /// batch-probe the filter with profile
  void contains_naive(u32 *__restrict__ keys, u32 key_cnt,
					  $u32 *__restrict__ bitmap, perf_data_t &perf_data) {
    std::cout << "naive kernel" << std::endl;
	// Allocate device memory and copy the keys to device
	$u32 *d_keys;
	u32 device_keys_size = sizeof($u32) * key_cnt;
	const std::size_t gran = 4ull * 1024;
	const std::size_t device_keys_alloc_size =
		((device_keys_size + (gran - 1)) / gran) * gran;
	cudaMalloc((void **)&d_keys, device_keys_alloc_size);
	cuda_check_error();
	cudaMemcpy(d_keys, keys, device_keys_size, cudaMemcpyHostToDevice);
	cuda_check_error();

	u64 repeats = 10;
	i32 block_size = 32;
	i32 block_cnt = (key_cnt + block_size - 1) / block_size;
	// Allocate memory for the result bitmap
	$u32 *device_bitmap;
	const std::size_t device_bitmap_size = (key_cnt + 7) / 8; // one bit per key
	cudaMalloc((void **)&device_bitmap, device_bitmap_size);
	cuda_check_error();

	// Probe
	i32 elements_per_thread = warp_size;
	i32 elements_per_block = block_size * elements_per_thread;
	i32 block_count = (key_cnt + elements_per_block - 1) / elements_per_block;
	perf_data.cuda_block_size = block_size;
	perf_data.cuda_block_cnt = block_cnt;

	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	std::cout << "Probing ..." << std::endl;
	if (repeats > 1) {
        contains_naive_kernel<<<block_count, block_size>>>(
		  filter, device_word_array, d_keys, key_cnt, device_bitmap);
    }
	// Real Experiment
	auto start_probe = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < repeats; i++) {
	  contains_naive_kernel<<<block_count, block_size>>>(
		  filter, device_word_array, d_keys, key_cnt, device_bitmap);
	}
	cudaDeviceSynchronize();
	cuda_check_error();
	auto end_probe = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration_probe = end_probe - start_probe;
	perf_data.probe_time = duration_probe.count() / repeats;
	perf_data.probes_per_second = static_cast<u64>(key_cnt * repeats / duration_probe.count());

	perf_data.total_throughput = static_cast<u64>((key_cnt) / perf_data.probe_time);

	std::cout << "Copying result back ..." << std::endl;
	// Copy back the resulting bitmap to the host memory
	cudaMemcpy(bitmap, device_bitmap, device_bitmap_size, cudaMemcpyDeviceToHost);
	cuda_check_error();
	cudaDeviceSynchronize();
	cuda_check_error();
	std::cout << "Copied ..." << std::endl;

	// Free temporary resources
	cudaDeviceSynchronize();
	cudaFree(d_keys);
	cudaFree(device_bitmap);
	cuda_check_error();
  }

  struct probe {
	/// reference to the CUDA filter instance
	const cuda_filter &cuda_filter_instance;

	/// the key type
	using key_t = typename filter_t::key_t;

	/// the CUDA stream to use
	const cudaStream_t &cuda_stream;

	/// CUDA events used for synchronization
	cudaEvent_t start_event;
	cudaEvent_t stop_event;

	/// the max batch size
	u64 batch_size;

	/// pointer to the keys on the device
	key_t *device_keys;

	/// pointer to the result bitmap on the device
	$u32 *device_bitmap;

	/// pointer to the result bitmap on the host
	$u64 *host_bitmap;

	/// c'tor
	probe(const cuda_filter &cuda_filter_instance, u64 batch_size,
		  const cudaStream_t &cuda_stream)
		: cuda_filter_instance(cuda_filter_instance),
		  batch_size(batch_size),
		  cuda_stream(cuda_stream) {
	  /// allocate device memory for the keys and for the result bitmap
	  cudaMalloc(&device_keys, batch_size * sizeof(key_t));
	  cuda_check_error();
	  cudaMalloc(&device_bitmap, batch_size / 8);
	  cuda_check_error();
	  cudaMallocHost(&host_bitmap, batch_size / 8);
	  cuda_check_error();

	  /// create events
	  cudaEventCreate(&start_event);
	  cudaEventCreate(&stop_event);
	}

	/// d'tor
	~probe() {
	  cudaFree(device_keys);
	  cudaFree(device_bitmap);
	  cudaFree(host_bitmap);
	  cudaEventDestroy(start_event);
	  cudaEventDestroy(stop_event);
	}

	/// asynchronously batch-probe the filter
	void contains(const key_t *keys, u32 key_cnt) {
	  // copy the keys to the pre-allocated device memory
	  cudaEventRecord(start_event, 0);
	  cuda_check_error();
	  cudaMemcpyAsync(device_keys, keys, batch_size * sizeof(key_t),
					  cudaMemcpyHostToDevice, cuda_stream);
	  cuda_check_error();
	  // copy back the result bitmap to pre-allocated host memory
	  cudaMemcpyAsync(host_bitmap, device_bitmap, batch_size / 8,
					  cudaMemcpyDeviceToHost, cuda_stream);
	  cuda_check_error();
	  cudaEventRecord(stop_event, 0);
	  cuda_check_error();
	}

	/// blocks until a asynchronously executed query is finished.
	void wait() {
	  cudaEventSynchronize(stop_event);
	  cuda_check_error();
	}
  };
};
//===----------------------------------------------------------------------===//
