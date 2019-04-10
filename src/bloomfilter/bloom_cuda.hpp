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
	FilterWrapper(size_t bloom_size, amsfilter::Config config) : bloom_filter(config, bloom_size) {}
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

	bool contains(const key_t key) {
		return bloom_filter.contains(key);
	}

	NO_INLINE int contains_sel(int* CPU_R res, int32_t* CPU_R keys,
			int* CPU_R sel, int num) {
		return Vectorized::select(res, sel, num, [&] (auto i) {
			return contains((key_t)keys[i]);
		});
	}

	NO_INLINE void contains_chr(uint8_t* CPU_R res, int32_t* CPU_R keys,
			int* CPU_R sel, int num) {
		Vectorized::map(sel, num, [&] (auto i) {
			res[i] = (uint8_t)contains((key_t)keys[i]);
		});
	}
	amsfilter::AmsFilterLite bloom_filter;
};
