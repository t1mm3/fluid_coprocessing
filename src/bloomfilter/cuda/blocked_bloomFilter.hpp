#include "../cuda_helper.hpp"
#include "kernels.hpp"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>

#define BEGIN_BIT 26
#define END_BIT 32

//===----------------------------------------------------------------------===//
// Predicate evaluation 
template <typename T> struct is_true : public thrust::unary_function<T, bool> {
	__host__ __device__ bool operator()(T x) {
		return x == 1;
	}
};

template <typename filter_t> struct cuda_filter {
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
	/// pointer to the keys on the device
	word_t *device_keys_array;
	/// size of the keys in bytes
	$u64 keys_array_size;

	/// c'tor
	cuda_filter(const filter_t &filter, const word_t *__restrict word_array, const std::size_t word_cnt)
	    : filter(filter) {
		cudaSetDevice(0);
		// copy the filter data to device memory
		assert(word_cnt > 0);
		word_array_size = word_cnt * sizeof(word_t);
		cudaMalloc((void **)&device_word_array, word_array_size);
		cudaMemcpy(device_word_array, word_array, word_array_size, cudaMemcpyHostToDevice);
	}
	//! Constructs a cuda filter allocating and copying both bloom filter and keys on GPU
	cuda_filter(const filter_t &filter, const word_t *__restrict word_array, const std::size_t word_cnt, const key_t *__restrict keys, const std::size_t key_cnt)
	    : filter(filter) {
		cudaSetDevice(0);
		// copy the filter data to device memory
		assert(word_cnt > 0);
		assert(key_cnt > 0);
		word_array_size = word_cnt * sizeof(word_t);
		keys_array_size = key_cnt * sizeof(key_t);
		cudaMalloc((void **)&device_word_array, word_array_size);
		cudaMalloc((void **)&device_keys_array, keys_array_size);
		cudaMemcpy(device_word_array, word_array, word_array_size, cudaMemcpyHostToDevice);
		cudaMemcpy(device_keys_array, keys, keys_array_size, cudaMemcpyHostToDevice);
	}

	/// d'tor
	~cuda_filter() {
		cudaFree(device_word_array);
		cuda_check_error();
	}
	/// batch-probe the filter
	void contains(u32 *__restrict__ keys, u32 key_cnt, $u32 *__restrict__ bitmap, $u32 *__restrict__ /*keys_hash*/) {
		perf_data_t perf_data;
		contains_naive(keys, key_cnt, bitmap, perf_data);
	}

	/// batch-probe the filter with profile
	void contains_clustering(u32 *__restrict__ keys, u32 key_cnt, $u32 *__restrict__ bitmap, perf_data_t &perf_data,
	                         std::size_t bits_to_sort) {
		// Allocate device memory and copy the keys to device
		$u32 *d_keys, *d_keys_hash, *d_positions;
		u32 device_keys_size = sizeof($u32) * key_cnt;
		const std::size_t gran = 4ull * 1024; // page allignment
		const std::size_t device_keys_alloc_size = ((device_keys_size + (gran - 1)) / gran) * gran;

		// Allocate device memory for hash_values and positions
		cudaMalloc((void **)&d_keys, 	  device_keys_alloc_size);
		cudaMalloc((void **)&d_keys_hash, device_keys_alloc_size);
		cudaMalloc((void **)&d_positions, device_keys_alloc_size);
		cuda_check_error();

		// Copy keys to device
		cudaMemcpy(d_keys, keys, device_keys_size, cudaMemcpyHostToDevice);

		// Calculate Hash Values
		i32 block_size = 32;
		i32 block_cnt = (key_cnt + block_size - 1) / block_size;
		auto start_hash = std::chrono::high_resolution_clock::now();

		calculate_hash_kernel<<<block_cnt, block_size>>>(filter, d_keys, key_cnt, d_keys_hash, d_positions);
		cudaDeviceSynchronize();

		auto end_hash = std::chrono::high_resolution_clock::now();
		perf_data.hash_time = std::chrono::duration<double>(end_hash - start_hash).count();
		perf_data.hash_throughput = static_cast<u64>(key_cnt / perf_data.hash_time);
		cudaFree(d_keys);

		// Sort Hash Values by 6 MSB
		// Allocate auxiliary buffers for sorted keys and positions
		$u32 *d_sorted_keys, *d_sorted_positions;
		cudaMalloc((void **)&d_sorted_keys, device_keys_alloc_size);
		cudaMalloc((void **)&d_sorted_positions, device_keys_alloc_size);
		cuda_check_error();

		// Sorting
		// Allocate temp storage for radix sort
		unsigned char *temp_storage;
		auto temp_storage_size = get_temp_storage_requirement<key_t, key_t>(key_cnt);
		cudaMalloc((void **)&temp_storage, temp_storage_size);

		auto start_sort = std::chrono::high_resolution_clock::now();

		auto status =
		    cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_size, d_keys_hash, d_sorted_keys, d_positions,
		                                    d_sorted_positions, key_cnt, (END_BIT - bits_to_sort), END_BIT);
		cudaDeviceSynchronize();

		auto end_sort = std::chrono::high_resolution_clock::now();
		cudaFree(d_keys_hash);
		cudaFree(d_positions);
		cudaFree(temp_storage);
		perf_data.sort_time = std::chrono::duration<double>(end_sort - start_sort).count();
		perf_data.sort_throughput = static_cast<u64>(key_cnt / perf_data.sort_time);

		// Allocate memory for the result bitmap
		$u32 *device_bitmap;
		const std::size_t device_bitmap_size = (key_cnt) * sizeof($u32); // one position per key
		cudaMalloc((void **)&device_bitmap, device_bitmap_size);
		cuda_check_error();

		// Probe with sorted hashes
		u64 repeats = 10;
		i32 elements_per_thread = warp_size;
		i32 elements_per_block = block_size * elements_per_thread;
		i32 block_count = (key_cnt + elements_per_block - 1) / elements_per_block;
		perf_data.cuda_block_size = block_size;
		perf_data.cuda_block_cnt = block_count;

		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		// warm up
		if (repeats > 1) {
			contains_clustered_kernel<<<block_count, block_size>>>(filter, device_word_array, d_sorted_keys,
			                                                       key_cnt, device_bitmap);
			cudaDeviceSynchronize();
			cuda_check_error();
		}
		// Real Experiment
		auto start_probe = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i < repeats; i++) {
			contains_clustered_kernel<<<block_count, block_size>>>(filter, device_word_array, d_sorted_keys,
			                                                       key_cnt, device_bitmap);
			cudaDeviceSynchronize();
			cuda_check_error();
		}

		auto end_probe = std::chrono::high_resolution_clock::now();
		perf_data.probe_time = std::chrono::duration<double>(end_probe - start_probe).count() / repeats;
		perf_data.probes_per_second = static_cast<u64>((key_cnt) / perf_data.probe_time);

		uint32_t *result_bitmap;
		cudaMalloc((void **)&result_bitmap, device_bitmap_size);

		auto start_candidates = std::chrono::high_resolution_clock::now();
		auto output_end = thrust::copy_if(thrust::device, d_sorted_positions, d_sorted_positions + key_cnt,
		                                  device_bitmap, result_bitmap, is_true<uint32_t>());

		auto end_candidates = std::chrono::high_resolution_clock::now();
		perf_data.candidate_time = std::chrono::duration<double>(end_candidates - start_candidates).count();
		double total_time = static_cast<double>(perf_data.hash_time + perf_data.sort_time + perf_data.probe_time +
		                                        perf_data.candidate_time);
		perf_data.total_throughput = static_cast<u64>((key_cnt) / total_time);

		// copy back only the candidate list
		size_t result_size = (output_end - result_bitmap) * sizeof(uint32_t);
		assert(result_size <= device_bitmap_size);
		cudaMemcpy(bitmap, result_bitmap, result_size, cudaMemcpyDeviceToHost);
		cuda_check_error();
		cudaDeviceSynchronize();
		cuda_check_error();

		// Free temporary resources
		cudaFree(d_sorted_keys);
		cudaFree(d_sorted_positions);
		cudaFree(device_bitmap);
		cudaFree(result_bitmap);
		cuda_check_error();
	}

	/// batch-probe the filter with profile
	void contains_naive(u32 *__restrict__ keys, u32 key_cnt, $u32 *__restrict__ bitmap, perf_data_t &perf_data) {
		std::cout << "naive kernel" << std::endl;
		// Allocate device memory and copy the keys to device
		$u32 *d_keys;
		u32 device_keys_size = sizeof($u32) * key_cnt;
		const std::size_t gran = 4ull * 1024;
		const std::size_t device_keys_alloc_size = ((device_keys_size + (gran - 1)) / gran) * gran;
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
			contains_naive_kernel<<<block_count, block_size>>>(filter, device_word_array, d_keys, key_cnt,
			                                                   device_bitmap);
		}
		// Real Experiment
		auto start_probe = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i < repeats; i++) {
			contains_naive_kernel<<<block_count, block_size>>>(filter, device_word_array, d_keys, key_cnt,
			                                                   device_bitmap);
			cudaDeviceSynchronize();
		}
		auto end_probe = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> duration_probe = end_probe - start_probe;
		perf_data.probe_time = duration_probe.count() / repeats;
		perf_data.probes_per_second = static_cast<u64>((key_cnt * repeats) / duration_probe.count());

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
	//! batch_probe for co-processing
	void contains_baseline(u32 *__restrict__ d_keys, int64_t key_cnt, $u32 *__restrict__ device_bitmap) {
	
		i32 block_size = 32;
		// Probe
		i32 elements_per_thread = warp_size;
		i32 elements_per_block = block_size * elements_per_thread;
		i32 block_count = (key_cnt + elements_per_block - 1) / elements_per_block;

		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		// Real Experiment
		contains_naive_kernel<<<block_count, block_size>>>(filter, device_word_array, d_keys, key_cnt,
			                                                device_bitmap);
		cudaDeviceSynchronize();
	}

	//! batch_probe for co-processing
	void contains_with_sorting(u32 *__restrict__ d_keys, int64_t key_cnt, $u32 *__restrict__ device_bitmap,
	                    std::size_t& size_no_matches) {
		// Allocate device memory and copy the keys to device
		$u32 *d_keys_hash, *d_positions;
		std::size_t key_alloc_size = key_cnt * sizeof(u32);

		// Calculate Hash Values
		i32 block_size = 32;
		i32 block_cnt = (key_cnt + block_size - 1) / block_size;
		calculate_hash_kernel<<<block_cnt, block_size>>>(filter, d_keys, key_cnt, d_keys_hash, d_positions);
		cudaDeviceSynchronize();

		// Sort Hash Values by 6 MSB
		$u32 *d_sorted_keys, *d_sorted_positions;
		cudaMalloc((void **)&d_sorted_keys, key_alloc_size);
		cudaMalloc((void **)&d_sorted_positions, key_alloc_size);
		cuda_check_error();

		// Sorting
		auto temp_storage_size = get_temp_storage_requirement<key_t, key_t>(key_cnt);
		unsigned char *temp_storage;
		cudaMalloc((void **)&temp_storage, temp_storage_size);
		auto status =
		    cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_size, d_keys_hash, d_sorted_keys, d_positions,
		                                    d_sorted_positions, key_cnt, (END_BIT - BEGIN_BIT), END_BIT);
		cudaDeviceSynchronize();
		cudaFree(d_keys_hash);
		cudaFree(d_positions);
		cudaFree(temp_storage);

		// Probe with sorted hashes
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		contains_clustered_kernel<<<block_cnt, block_size>>>(filter, device_word_array, d_sorted_keys,
			                                                    key_cnt, device_bitmap);
		cudaDeviceSynchronize();
		cuda_check_error();
		uint32_t *result_bitmap;
		cudaMalloc((void **)&result_bitmap, key_alloc_size);
		auto output_end = thrust::copy_if(thrust::device, d_sorted_positions, d_sorted_positions + key_cnt,
		                                  device_bitmap, result_bitmap, is_true<uint32_t>());

		// copy back only the candidate list
		size_t result_size = (output_end - result_bitmap) * sizeof(uint32_t);
		assert(result_size <= key_alloc_size);
		size_no_matches = result_size;

		// Free temporary resources
		cudaDeviceSynchronize();
		cudaFree(d_sorted_keys);
		cudaFree(d_sorted_positions);
		cudaFree(result_bitmap);
		cuda_check_error();
	}

	void contains_with_keys_on_gpu(u32 offset, int64_t key_cnt, $u32 *__restrict__ device_bitmap) {
	
		i32 block_size = 32;
		// Probe
		i32 elements_per_thread = warp_size;
		i32 elements_per_block = block_size * elements_per_thread;
		i32 block_count = (key_cnt + elements_per_block - 1) / elements_per_block;

		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		// Real Experiment
		contains_naive_kernel<<<block_count, block_size>>>(filter, device_word_array, &device_keys_array[offset], key_cnt,
			                                                device_bitmap);
		cudaDeviceSynchronize();
	}

	struct probe {
		/// reference to the CUDA filter instance
		cuda_filter &cuda_filter_instance;

		/// the key type
		using key_t = typename filter_t::key_t;

		/// the CUDA stream to use
		const cudaStream_t &cuda_stream;

	  	/// CUDA device
	  	u32 device_no_;

		/// CUDA events used for synchronization
		cudaEvent_t start_event;
		cudaEvent_t stop_event;

		/// the max batch size
		u64 batch_size;

		/// pointer to the keys on the device
		key_t *device_in_keys;

		/// pointer to the result bitmap on the device
		$u32 *device_bitmap;

		/// pointer to the result bitmap on the host
		$u32 *host_bitmap;

		/// c'tor
		probe(cuda_filter &cuda_filter_instance, u64 batch_size, const cudaStream_t &cuda_stream, u32 cuda_device_no)
		    : cuda_filter_instance(cuda_filter_instance), batch_size(batch_size), cuda_stream(cuda_stream), device_no_(cuda_device_no) {
		    assert(batch_size > 0);
			cudaSetDevice(device_no_);
			// Allocate device memory for the keys and for the result bitmap
			cudaMalloc((void **)&device_in_keys, batch_size * sizeof(key_t));
			assert(batch_size % 32 == 0);
			cudaMalloc((void **)&device_bitmap, batch_size / 8);
			// Allocate host memory for result bitmap
			host_bitmap = nullptr;
			cudaMallocHost((void **)&host_bitmap, batch_size / 8, cudaHostAllocPortable);
			// memset(host_bitmap, 0, batch_size / 8);
			assert(host_bitmap != nullptr);

			/// create events
			cudaEventCreate(&start_event);
			cudaEventCreate(&stop_event);
		}

		/// d'tor
		~probe() {
			cudaFree(device_in_keys);
			cudaFree(device_bitmap);
			cudaFreeHost(host_bitmap);
			cudaEventDestroy(start_event);
			cudaEventDestroy(stop_event);
		}

		/// asynchronously batch-probe the filter
		void contains(const key_t *keys, int64_t key_cnt) {
			cudaSetDevice(device_no_);
			// copy the keys to the pre-allocated device memory
			assert(key_cnt > 0);
			assert(device_in_keys != nullptr);
			cudaEventRecord(start_event, 0);
			cudaMemcpyAsync(device_in_keys, keys, key_cnt * sizeof(key_t), cudaMemcpyHostToDevice, cuda_stream);
			cuda_filter_instance.contains_baseline(&device_in_keys[0], key_cnt, &device_bitmap[0]);
			// copy back the result bitmap to pre-allocated host memory
			cudaMemcpyAsync(host_bitmap, device_bitmap, key_cnt / 8, cudaMemcpyDeviceToHost, cuda_stream);
			cudaEventRecord(stop_event, 0);
		}


		/// asynchronously batch-probe the filter
		void contains_sort(const key_t *keys, int64_t key_cnt) {
			cudaSetDevice(device_no_);
			// copy the keys to the pre-allocated device memory
			assert(key_cnt > 0);
			assert(device_in_keys != nullptr);
			cudaEventRecord(start_event, 0);
			cudaMemcpyAsync(device_in_keys, keys, key_cnt * sizeof(key_t), cudaMemcpyHostToDevice, cuda_stream);
			size_t no_matches_size = 0;
			cuda_filter_instance.contains_with_sorting(&device_in_keys[0], key_cnt, &device_bitmap[0], &no_matches_size);
			// copy back the result bitmap to pre-allocated host memory
			cudaMemcpyAsync(host_bitmap, device_bitmap, no_matches_size, cudaMemcpyDeviceToHost, cuda_stream);
			cudaEventRecord(stop_event, 0);
		}

		/// asynchronously batch-probe the filter
		void contains_in_gpu_data(int64_t key_cnt, int64_t offset) {
			cudaSetDevice(device_no_);
			// copy the keys to the pre-allocated device memory
			assert(key_cnt > 0);
			assert(offset > 0);
			cudaEventRecord(start_event, 0);
			cuda_filter_instance.contains_with_keys_on_gpu(offset, key_cnt, &device_bitmap[0]);
			// copy back the result bitmap to pre-allocated host memory
			cudaMemcpyAsync(host_bitmap, device_bitmap, key_cnt / 8, cudaMemcpyDeviceToHost, cuda_stream);
			cudaEventRecord(stop_event, 0);
		}

		/// blocks until a asynchronously executed query is finished.
		void wait() {
			cudaEventSynchronize(stop_event);
		}
		// checks whether the gpu has finished
		bool is_done() {
    		return cudaEventQuery(stop_event) == cudaSuccess;
  		}

  		$u32* get_results() {
    		return host_bitmap;
  		}
	};
};
//===----------------------------------------------------------------------===//
