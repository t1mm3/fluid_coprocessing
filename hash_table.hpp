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
	void** heads;

	uint32_t mod_mask;

	size_t bucket_size;
	size_t next_offset;
	size_t hash_offset;

	std::atomic<int64_t> num_buckets;
	size_t max_buckets;

	char* value_space;

	static void** BucketPtrOfNext(void* bucket, size_t offset) {
		return (void**)((char**)bucket + offset);
	}

	static uint32_t* BucketPtrOfHash(void* bucket, size_t offset) {
		return (uint32_t*)((char**)bucket + offset);
	}

	static void BucketSetNext(void* bucket, void* next, size_t offset) {
		auto pnext = BucketPtrOfNext(bucket, offset);
		*pnext = next;
	}

	static void* BucketGetNext(void* bucket, size_t offset) {
		return *BucketPtrOfNext(bucket, offset);
	}

	static void _GetNext(void** R out, void** R buckets, size_t next_offset,
			int* R sel, int num);
	static void _Insert(void** R buckets, void** R heads, uint32_t* R hash,
			uint32_t mod_mask, size_t next_offset, int* R sel, int num);

	static void BucketInit(void** R buckets, void** R heads, uint32_t* R hash,
			uint32_t mod_mask, int* R sel, int num);

	template<typename CHECK_KEYS>
	void _Probe(void** tmp_buckets, int* tmp_sel,
			bool* match, uint32_t* hash,
			int* in_sel, int in_num,
			CHECK_KEYS&& check) {

		assert(heads);
		size_t n = in_num;
		int* sel = in_sel;

		// Init using heads
		BucketInit(tmp_buckets, heads, hash, mod_mask, sel, n);

		Vectorized::map_not_match_void(match, tmp_buckets, NULL, sel, n);

		n = Vectorized::select_match(tmp_sel, match, sel, n);
		sel = tmp_sel;

		// continue for matches
		while (n) {
			// check keys
			check(match, tmp_buckets, sel, n);
			n = Vectorized::select_not_match(tmp_sel, match, sel, n);

			_GetNext(tmp_buckets, tmp_buckets, next_offset, sel, n);

			Vectorized::map_not_match_void(match, tmp_buckets, NULL, sel, n);
			n = Vectorized::select_match(tmp_sel, match, sel, n);
		};
	}

	std::mutex finalize_build_mutex;

public:
	HashTablinho(size_t bsize, size_t capacity) {
		assert(bsize > sizeof(int64_t));
		bucket_size = bsize + sizeof(void*) + sizeof(uint32_t);

		hash_offset = bsize;
		next_offset = bsize + sizeof(uint32_t);

		heads = nullptr;
		mod_mask = 0;
		num_buckets = 0;

		max_buckets = std::max(capacity, (size_t)1024);
		value_space = new char[bucket_size * max_buckets];
	}

	~HashTablinho() {
		delete[] value_space;
		if (heads) {
			delete[] heads;
		}
	}

	void Insert(int32_t* key, uint32_t* hash, int* sel, int num);
	void FinalizeBuild();

	struct ProbeContext {
		bool* matches;
		void** tmp_buckets;
		int* tmp_sel;
	};

	template<size_t VSIZE>
	struct StaticProbeContext : ProbeContext {
	private:
		bool amatches[VSIZE];

		void* atmp_buckets[VSIZE];
		int atmp_sel[VSIZE];
	public:
		StaticProbeContext() : ProbeContext() {
			matches = &amatches[0];
			tmp_buckets = &atmp_buckets[0];
			tmp_sel = &atmp_sel[0];
		}
	};

	void Probe(ProbeContext &ctx, bool* matches, int32_t* keys, uint32_t* hash,
			int *sel, int num) {
		if (!heads) {
			FinalizeBuild();
		}
		_Probe(ctx.tmp_buckets, ctx.tmp_sel, matches, hash, sel, num,
			[&] (auto& match, auto& buckets, auto& sel, auto& n) {
				Vectorized::check_ptr<int32_t>(match, keys, (int32_t**)buckets, sel, n);
		});
	}

	// Gathers payload column, supposed to be run after Probe for fetching the buckets
	// However, the selection vector should only contain matching keys
	void ProbeGather(ProbeContext &ctx, int32_t* coldata, int64_t colidx, int* sel, int n) {
		Vectorized::gather_ptr<int32_t>(coldata, (int32_t**)ctx.tmp_buckets, colidx, sel, n);
	}
};
