#pragma once

#include "bloom_cuda_filter_template.hpp"
#include "cuda_helper.hpp"

#include <cub/thread/thread_load.cuh>
#include <cuda_runtime.h>
#include <dtl/dtl.hpp>
#include <dtl/env.hpp>

//===----------------------------------------------------------------------===//
// Helper to determine the largest native type suitable for data movement.
template <u32 size_in_bytes> struct largest_native_type_helper {
	using type = typename largest_native_type_helper<size_in_bytes / 2>::type;
};
template <> struct largest_native_type_helper<16> { using type = int4; };
template <> struct largest_native_type_helper<8> { using type = $u64; };
template <> struct largest_native_type_helper<4> { using type = $u32; };
template <> struct largest_native_type_helper<2> { using type = $u16; };
template <> struct largest_native_type_helper<1> { using type = $u8; };
/// Determine the largest native type suitable for data movement.
template <typename T, u32 cnt> struct largest_native_type {
	using type = typename largest_native_type_helper<sizeof(T) * cnt>::type;
};
//===----------------------------------------------------------------------===//
// Gathers consecutive items from memory.
template <typename T, u32 items_per_thread, cub::CacheLoadModifier load_modifier = cub::LOAD_CA>
struct block_gather_internal {
	__device__ __forceinline__ static void load(const T *block_ptr, T *shared_mem_out) {
		__shared__ const T *block_ptrs[warp_size];
		u32 lid = warp_local_thread_id();
		// Write the block ptrs of all threads to shared memory.
		block_ptrs[lid] = block_ptr;
		__syncwarp();
// Copy the block to shared memory.
#pragma unroll
		for ($u32 l = 0; l < items_per_thread; ++l) {
			u32 j = (warp_size * l) + lid;
			u32 p = j / items_per_thread;
			u32 o = j % items_per_thread;
			const T *ptr = block_ptrs[p];
			shared_mem_out[j] = cub::ThreadLoad<cub::LOAD_CA>(&ptr[o]);
		}
		__syncwarp();
	}
};
/// Gathers consecutive items from memory.
template <typename T, u32 items_per_thread, cub::CacheLoadModifier load_modifier = cub::LOAD_DEFAULT>
struct block_gather {
	__device__ __forceinline__ static void load(const T *block_ptr, T *shared_mem_out) {
		// Determine the data movement type.
		using Tm = typename largest_native_type<T, items_per_thread>::type;
		static constexpr u32 move_cnt = items_per_thread / (sizeof(Tm) / sizeof(T));
		block_gather_internal<Tm, move_cnt, load_modifier>::load(reinterpret_cast<const Tm *>(block_ptr),
		                                                         reinterpret_cast<Tm *>(shared_mem_out));
	}
};
//===----------------------------------------------------------------------===//
/// A straight-forward kernel to probe a Bloom filter.
template <typename filter_t>
__global__ void contains_kernel(const filter_t filter, const typename filter_t::word_t *__restrict__ word_array,
                                u32 *__restrict__ keys, u32 key_cnt, $u32 *__restrict__ result_bitmap) {

	// Who am I?
	u32 wid = global_warp_id();
	u32 lid = warp_local_thread_id();

	// The output this executing thread will write later on (until then, the
	// result is kept in a register).
	$u32 thread_local_bitmap = 0u;

	constexpr u32 elements_per_thread = warp_size; // ... processed sequentially
	constexpr u32 elements_per_warp = elements_per_thread * warp_size;

	// Where to start reading the input?
	$u32 read_pos = wid * elements_per_warp + lid;

	// Each thread processes multiple elements sequentially.
	for ($u32 i = 0; i != elements_per_thread; ++i) {
		auto is_contained = (read_pos < key_cnt) ? filter.contains(word_array, keys[read_pos]) : false;
		u32 bitmap = __ballot_sync(0xffffffff, is_contained);
		thread_local_bitmap = (lid == i) ? bitmap : thread_local_bitmap;
		read_pos += warp_size;
	}
	__syncwarp();
	// Every thread writes a single word of the output bitmap.
	u32 write_pos = global_thread_id();
	result_bitmap[write_pos] = thread_local_bitmap;
}
//===----------------------------------------------------------------------===//
/// Similar to the straight-forward kernel above, but the blocks are explicitly
/// copied to shared memory before they are probed.
///
/// Note that this kernel only supports a fixed block size, which is 32.
template <typename filter_t>
__global__ void
contains_kernel_with_block_prefetch(const filter_t filter, const typename filter_t::word_t *__restrict__ word_array,
                                    u32 *__restrict__ keys, u32 key_cnt, $u32 *__restrict__ result_bitmap) {

	using word_t = typename filter_t::word_t;
	static constexpr u32 word_cnt_per_block = filter_t::word_cnt_per_block;

	__shared__ word_t block_cache[word_cnt_per_block * warp_size];

	// Who am I?
	u32 wid = global_warp_id();
	u32 lid = warp_local_thread_id();

	word_t *block_cache_ptr = &block_cache[word_cnt_per_block * lid];

	// The output this executing thread will write later on (until then, the
	// result is kept in a register).
	$u32 thread_local_bitmap = 0u;

	constexpr u32 elements_per_thread = warp_size; // ... processed sequentially
	constexpr u32 elements_per_warp = elements_per_thread * warp_size;

	// Where to start reading the input?
	$u32 read_pos = wid * elements_per_warp + lid;

	// Each thread processes multiple elements sequentially.
	for ($u32 i = 0; i != elements_per_thread; ++i) {
		auto is_contained = false;
		if (read_pos < key_cnt) {
			const auto key = keys[read_pos];
			const auto block_idx = filter.get_block_idx(key);
			const auto block_ptr = word_array + (word_cnt_per_block * block_idx);
			block_gather<word_t, word_cnt_per_block>::load(block_ptr, block_cache);
			is_contained = filter_t::block_t::contains(block_cache_ptr, key);
		}
		u32 bitmap = __ballot_sync(0xffffffff, is_contained);
		thread_local_bitmap = (lid == i) ? bitmap : thread_local_bitmap;
		read_pos += warp_size;
	}
	__syncwarp();
	// Every thread writes a single word of the output bitmap.
	u32 write_pos = global_thread_id();
	result_bitmap[write_pos] = thread_local_bitmap;
}
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
template <typename filter_t> struct cuda_filter {

	struct perf_data_t {
		$i32 cuda_block_size = 0;
		$i32 cuda_block_cnt = 0;

		$u64 probes_per_second = 0;
	};

	using key_t = typename filter_t::key_t;
	using word_t = typename filter_t::word_t;

	/// Instance of the host-side filter.
	const filter_t &filter;
	/// Pointer to the filter data on the device.
	word_t *device_word_array;
	/// Size of the filter data in bytes.
	$u64 word_array_size;

	/// C'tor. Copies the filter to the device memory.
	cuda_filter(const filter_t &filter, const word_t *__restrict word_array, const std::size_t word_cnt)
	    : filter(filter) {
		// Copy the filter data to device memory.
		word_array_size = word_cnt * sizeof(word_t);
		cudaMalloc((void **)&device_word_array, word_array_size);
		cuda_check_error();
		cudaMemcpy(device_word_array, word_array, word_array_size, cudaMemcpyHostToDevice);
		cuda_check_error();
	};

	/// D'tor.
	~cuda_filter() {
		cudaFree(device_word_array);
		cuda_check_error();
	};

	/// Batch-probe the filter.
	void contains(u32 *__restrict keys, u32 key_cnt, $u64 *__restrict bitmap) {
		perf_data_t perf_data;
		contains(keys, key_cnt, bitmap, perf_data);
	}

	/// Batch-probe the filter.
	void contains(u32 *__restrict keys, u32 key_cnt, $u64 *__restrict bitmap, perf_data_t &perf_data) {
		// Allocate device memory and copy the keys.
		$u32 *device_keys;
		u32 device_keys_size = sizeof($u32) * key_cnt;
		cudaError_t cu_err;
		const std::size_t gran = 4ull * 1024;
		const std::size_t device_keys_alloc_size = ((device_keys_size + (gran - 1)) / gran) * gran;
		cu_err = cudaMalloc((void **)&device_keys, device_keys_alloc_size);
		if (cu_err != cudaSuccess)
			std::cerr << "allocation failed" << std::endl;
		cuda_check_error();
		cu_err = cudaMemcpy(device_keys, keys, device_keys_size, cudaMemcpyHostToDevice);
		cuda_check_error();

		// Allocate memory for the result bitmap.
		$u32 *device_bitmap;
		u32 device_bitmap_size = (key_cnt + 7) / 8;
		cu_err = cudaMalloc((void **)&device_bitmap, device_bitmap_size);
		if (cu_err != cudaSuccess)
			std::cerr << "allocation failed" << std::endl;
		cuda_check_error();

		i32 block_size = dtl::env<$i32>::get("BLOCK_SIZE", warp_size);
		perf_data.cuda_block_size = block_size;

		// ---
		i32 elements_per_thread = warp_size;
		i32 elements_per_block = block_size * elements_per_thread;
		i32 block_cnt = (key_cnt + elements_per_block - 1) / elements_per_block;
		perf_data.cuda_block_cnt = block_cnt;

		cu_err = cudaDeviceSynchronize();
		cu_err = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		u64 repeats = 100;
		if (repeats > 1) {
			// warm up
			//      contains_kernel<<<block_cnt, block_size>>>(
			contains_kernel_with_block_prefetch<<<block_cnt, block_size>>>(filter, device_word_array, device_keys,
			                                                               key_cnt, device_bitmap);
			cu_err = cudaDeviceSynchronize();
			cuda_check_error();
		}
		auto start = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i < repeats; i++) {
			//      contains_kernel<<<block_cnt, block_size>>>(
			contains_kernel_with_block_prefetch<<<block_cnt, block_size>>>(filter, device_word_array, device_keys,
			                                                               key_cnt, device_bitmap);
			cu_err = cudaDeviceSynchronize();
		}
		cuda_check_error();
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> duration = end - start;
		u64 probes_per_second = static_cast<u64>(key_cnt * repeats / duration.count());

		perf_data.probes_per_second = probes_per_second;

		// Copy back the resulting bitmap to the host memory.
		cu_err = cudaMemcpy(bitmap, device_bitmap, device_bitmap_size, cudaMemcpyDeviceToHost);
		cuda_check_error();
		cu_err = cudaDeviceSynchronize();
		cuda_check_error();
		// free temporary resources
		cudaDeviceSynchronize();
		cu_err = cudaFree(device_keys);
		cu_err = cudaFree(device_bitmap);
	}

	struct probe {

		/// Reference to the CUDA filter instance.
		const cuda_filter &cuda_filter_instance;

		/// The key type.
		using key_t = typename filter_t::key_t;

		/// The CUDA stream to use.
		const cudaStream_t &cuda_stream;

		/// CUDA events used for synchronization. - We use pointers to the CUDA
		/// events to make probe instance move-constructable.
		cudaEvent_t *start_event;
		cudaEvent_t *stop_event;

		/// The max batch size.
		u64 batch_size;

		/// Pointer to the keys on the device.
		key_t *device_keys;

		/// Pointer to the result bitmap on the device.
		$u32 *device_bitmap;

		/// Pointer to the result bitmap on the host.
		$u64 *host_bitmap;

		/// The size of the result bitmap in bytes. Note that the size is a multiple
		/// eight bytes.
		$u64 bitmap_size_bytes;

		/// C'tor.
		probe(const cuda_filter &cuda_filter_instance, u64 batch_size, const cudaStream_t &cuda_stream)
		    : cuda_filter_instance(cuda_filter_instance), batch_size(batch_size), cuda_stream(cuda_stream),
		      bitmap_size_bytes((((batch_size + 63) / 64) * 64) / 8) {

			/// Allocate device memory for the keys and for the result bitmap.
			cudaMalloc(&device_keys, batch_size * sizeof(key_t));
			cuda_check_error();
			cudaMalloc(&device_bitmap, bitmap_size_bytes);
			cuda_check_error();
			cudaMallocHost(&host_bitmap, bitmap_size_bytes);
			cuda_check_error();

			/// Create events.
			start_event = new cudaEvent_t;
			cudaEventCreate(start_event);
			stop_event = new cudaEvent_t;
			cudaEventCreate(stop_event);
		}

		probe(probe &&src) noexcept
		    : cuda_filter_instance(src.cuda_filter_instance), cuda_stream(src.cuda_stream), start_event(nullptr),
		      stop_event(nullptr), batch_size(src.batch_size), device_keys(nullptr), device_bitmap(nullptr),
		      host_bitmap(nullptr), bitmap_size_bytes(0) {
			std::swap(start_event, src.start_event);
			std::swap(stop_event, src.stop_event);
			std::swap(device_keys, src.device_keys);
			std::swap(device_bitmap, src.device_bitmap);
			std::swap(host_bitmap, src.host_bitmap);
			std::swap(bitmap_size_bytes, src.bitmap_size_bytes);
		};
		probe(const probe &other) = delete;
		probe &operator=(const probe &other) = delete;
		probe &operator=(probe &&other) = delete;

		/// D'tor.
		~probe() {
			if (device_keys != nullptr) {
				cudaFree(device_keys);
				cuda_check_error();
			}
			if (device_bitmap != nullptr) {
				cudaFree(device_bitmap);
				cuda_check_error();
			}
			if (host_bitmap != nullptr) {
				cudaFreeHost(host_bitmap);
				cuda_check_error();
			}
			if (start_event != nullptr) {
				cudaEventDestroy(*start_event);
				cuda_check_error();
				delete start_event;
			}
			if (stop_event != nullptr) {
				cudaEventDestroy(*stop_event);
				cuda_check_error();
				delete stop_event;
			}
		}

		/// Asynchronously batch-probe the filter.
		void __attribute__((noinline)) contains(const key_t *keys, const std::size_t key_cnt) {
			if (key_cnt == 0)
				return;
			if (key_cnt > batch_size) {
				throw std::invalid_argument("The 'key_cnt' argument must not exceed"
				                            " the max. batch size.");
			}
			// Copy the keys to the pre-allocated device memory.
			cudaEventRecord(*start_event, cuda_stream);
			cuda_check_error();
			cudaMemcpyAsync(device_keys, keys, key_cnt * sizeof(key_t), cudaMemcpyHostToDevice, cuda_stream);
			cuda_check_error();

			i32 block_size = dtl::env<$i32>::get("BLOCK_SIZE", warp_size);
			//      perf_data.cuda_block_size = block_size;

			// ---
			i32 elements_per_thread = warp_size;
			i32 elements_per_block = block_size * elements_per_thread;
			i32 block_cnt = (key_cnt + elements_per_block - 1) / elements_per_block;
			//      perf_data.cuda_block_cnt = block_cnt;
			// Execute the kernel.
			contains_kernel_with_block_prefetch<<<block_cnt, block_size, 0, cuda_stream>>>(
			    cuda_filter_instance.filter, cuda_filter_instance.device_word_array, device_keys, key_cnt,
			    device_bitmap);
			// Copy back the result bitmap to pre-allocated host memory.
			cudaMemcpyAsync(host_bitmap, device_bitmap, bitmap_size_bytes, cudaMemcpyDeviceToHost, cuda_stream);
			cuda_check_error();
			cudaEventRecord(*stop_event, cuda_stream);
			cuda_check_error();
		}

		u1 __forceinline__ is_done() {
			return cudaEventQuery(*stop_event) == cudaSuccess;
		}

		/// Blocks until a asynchronously executed probe is finished.
		void __forceinline__ wait() {
			cudaEventSynchronize(*stop_event);
			cuda_check_error();
		}

		struct result_view {
			$u64 *bitmap_begin;
			$u64 *bitmap_end;

			u1 __forceinline__ is_match(std::size_t key_idx) const noexcept {
				const std::size_t word_idx = key_idx / 64;
				const std::size_t bit_idx = key_idx % 64;
				return ((bitmap_begin[word_idx] >> bit_idx) & 1) == 1;
			}
		};

		result_view results() {
			result_view r;
			r.bitmap_begin = host_bitmap;
			r.bitmap_end = host_bitmap + (bitmap_size_bytes / 8);
			return r;
		}

		using result_t = result_view;
	};
};
//===----------------------------------------------------------------------===//
