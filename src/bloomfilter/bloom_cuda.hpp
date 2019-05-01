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
#include <amsfilter/amsfilter_lite.hpp>
#include <amsfilter/internal/blocked_bloomfilter_template.hpp>


class FilterWrapper {
public:
	FilterWrapper(size_t bloom_size, amsfilter::Config config) : 
		bloom_filter(config, bloom_size), 
		filter_data(bloom_filter.size(), 0) {
	}
	~FilterWrapper() {}

	// void insert_with_hash(const key_t key) noexcept {
	// 	std::cout << "Addr " << &filter_data[0] << std::endl;
	// 	bloom_filter.insert_hash(&filter_data[0], key);
	// }

	void insert(const key_t key) noexcept {
		bloom_filter.insert(key);
	}

	// bool contains_with_hash(const key_t key) {
	// 	auto hash_key = bloom_filter.hash(key);
	// 	return bloom_filter.contains_with_hash(&filter_data[0], hash_key, key);
	// }

	void cache_keys(uint32_t* keys, size_t key_cnt) {
		auto keys_size = key_cnt * sizeof(uint32_t);
		cudaMalloc((void**)&device_keys, keys_size);
		cudaMemcpy(device_keys, keys, keys_size, cudaMemcpyHostToDevice);
		cuda_check_error();
	}

	bool contains(const key_t key) {
		return bloom_filter.contains(key);
	}

	amsfilter::AmsFilterLite bloom_filter;
	std::vector<amsfilter::word_t> filter_data;
	key_t* device_keys;
};
