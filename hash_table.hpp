/* Copyright (c) 2019 by Tim Gubner, CWI 
 * Licensed under GPLv3
 */

#pragma once
#include "build.hpp"
#include "vectorized.hpp"
#include <string>
#include <atomic>
#include <cassert>
#include <cstring>
#include <mutex>

struct HashTablinho {
private:
	bucket_t* heads;

	uint32_t mod_mask;

	size_t bucket_size;


	bucket_t* next_vector;
	size_t next_stride;

	uint32_t* hash_vector;
	size_t hash_stride;

	int32_t* key_vector;
	size_t key_stride;

	std::atomic<int64_t> num_buckets;
	size_t max_buckets;

	char* value_space;

	static void _Insert(bucket_t* R buckets, bucket_t* R heads, uint32_t* R hash,
			uint32_t mod_mask, bucket_t* R next_vector, size_t next_stride,
			int* R sel, int num);

	static void BucketInit(bucket_t* R buckets, bucket_t* R heads, uint32_t* R hash,
			uint32_t mod_mask, int* R sel, int num);

	std::mutex finalize_build_mutex;

public:
	HashTablinho(size_t bsize, size_t capacity);
	~HashTablinho();
	void Insert(int32_t* key, uint32_t* hash, int* sel, int num);
	void FinalizeBuild();

	struct ProbeContext {
		bool* matches;
		bucket_t* tmp_buckets;
		int* tmp_sel;
	};

	template<size_t VSIZE>
	struct StaticProbeContext : ProbeContext {
	private:
		bool amatches[VSIZE];
		bucket_t atmp_buckets[VSIZE];
		int atmp_sel[VSIZE];
	public:
		StaticProbeContext() : ProbeContext() {
			matches = &amatches[0];
			tmp_buckets = &atmp_buckets[0];
			tmp_sel = &atmp_sel[0];
		}
	};

	void Probe(ProbeContext &ctx, bool* matches, int32_t* keys, uint32_t* hash,
			int *sel, int num);

	// Gathers payload column, supposed to be run after Probe for fetching the buckets
	// However, the selection vector should only contain matching keys
	void ProbeGather(ProbeContext &ctx, int32_t* coldata, int64_t colidx, int* sel, int n);
};
