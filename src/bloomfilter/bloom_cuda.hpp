#pragma once
#include "cuda/blocked_bloomFilter.hpp"
#include "cuda_helper.hpp"
#include "util.hpp"

#include <algorithm>
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
#include <dtl/dtl.hpp>
#include <dtl/env.hpp>
#include <dtl/filter/blocked_bloomfilter/zoned_blocked_bloomfilter.hpp>
#include <dtl/hash.hpp>
#include <dtl/mem.hpp>
#include <dtl/thread.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

constexpr size_t DEFAULT_BLOOM_SIZE = (64 * 1024 * 1024 * 8);
//===----------------------------------------------------------------------===//
// Typedefs. (cache-sectorized blocked bloom filter)
using filter_key_t = $u32;
using hash_value_t = $u32;
using word_t = $u32;

// The first hash function to use inside the block. Note: 0 is used for block addressing
static constexpr u32 block_hash_fn_idx = 1;

// The block type.
template <u32 word_cnt, u32 zone_cnt, u32 k, u1 early_out = false>
using bbf_block_t = dtl::multizone_block<filter_key_t, word_t, word_cnt, zone_cnt, k, dtl::hasher, hash_value_t,
                                         block_hash_fn_idx, 0, zone_cnt, early_out>;

template <u32 word_cnt, u32 zone_cnt, u32 k, dtl::block_addressing addr = dtl::block_addressing::POWER_OF_TWO,
          u1 early_out = false>
using bbf_t = dtl::blocked_bloomfilter_logic<filter_key_t, dtl::hasher, bbf_block_t<word_cnt, zone_cnt, k, early_out>,
                                             addr, early_out>;

class FilterWrapper {
public:
	using filter_t = bbf_t<32, 1, 2>; // 128 bytes block 1 sector 1 zone 2 hash functions
	using word_t = typename filter_t::word_t;
	using cuda_filter_t = cuda_filter<filter_t>;
	using cuda_probe_t = typename cuda_filter<filter_t>::probe;
	using key_t = word_t;

	FilterWrapper(size_t bloom_size) : bloom_filter(bloom_size) {
		// set the device to be used for CUDA execution

		// Allocates Pinned memory on the host
		filter_data_size = (bloom_filter.word_cnt() + 1024) * sizeof(word_t);
		cudaError result_code = cudaHostAlloc((void **)&filter_data, filter_data_size, cudaHostAllocPortable);
		memset(filter_data, 0, filter_data_size);
		if (result_code != cudaSuccess)
			throw std::bad_alloc();
	}
	~FilterWrapper() {
		cudaError result_code = cudaFreeHost(filter_data);
		if (result_code != cudaSuccess)
			throw std::bad_alloc();
	}

	void insert_with_hash(const key_t key) noexcept {
		std::cout << "Addr " << &filter_data[0] << std::endl;
		bloom_filter.insert_hash(&filter_data[0], key);
	}

	void insert(const key_t key) noexcept {
		bloom_filter.insert(&filter_data[0], key);
	}

	bool contains_with_hash(const key_t key) {
		auto hash_key = bloom_filter.hash(key);
		return bloom_filter.contains_with_hash(&filter_data[0], hash_key);
	}

	bool contains(const key_t key) {
		return bloom_filter.contains(&filter_data[0], key);
	}

	filter_t bloom_filter;
	word_t *filter_data;
	size_t filter_data_size;
};
