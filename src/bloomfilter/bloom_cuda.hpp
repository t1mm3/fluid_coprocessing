#pragma once

#include "bloom_cuda_filter.hpp"
constexpr size_t DEFAULT_BLOOM_SIZE = (64 * 1024 * 1024 * 8);

class FilterWrapper {

	using filter_t = bbf_t<32, 1, 1, 2>; // 128 bytes block 1 sector 1 zone 2 hash functions
	using word_t = typename filter_t::word_t;
	using cuda_probe_t = typename cuda_filter<filter_t>::probe;
	using key_t = word_t;

	FilterWrapper(size_t bloom_size = DEFAULT_BLOOM_SIZE) : bloom_filter(bloom_size) {
		// array of bits
		word_t *filter_data;

		// Allocates Pinned memory on the host
		filter_data_size = (bloom_filter.word_cnt() + 1024) * sizeof(word_t);
		cudaMallocHost((void **)&filter_data, filter_data_size);
	}

	cuda_probe_t get_probe_instance(size_t batch_size, cudaStream_t &stream) {
		cuda_filter<filter_t> cuda_filter(bloom_filter, filter_data, filter_data_size);
		return cuda_probe_t(cuda_filter, batch_size, stream);
	}

	void insert_hash(word_t *__restrict filter_data, const key_t key) noexcept {
		auto hash_key = bloom_filter.hash(key);
		bloom_filter.insert_hash(&filter_data[0], hash_key);
	}

	void insert(word_t *__restrict filter_data, const key_t key) noexcept {
		bloom_filter.insert(&filter_data[0], key);
	}

	bool contains_with_hash(word_t *__restrict filter_data, const key_t key) {
		auto hash_key = bloom_filter.hash(key);
		return bloom_filter.contains_with_hash(&filter_data[0], hash_key);
	}

	bool contains(word_t *__restrict filter_data, const key_t key) {
		return bloom_filter.contains_with_hash(&filter_data[0], key);
	}

private:
	filter_t bloom_filter;
	word_t *filter_data;
	size_t filter_data_size;
};
