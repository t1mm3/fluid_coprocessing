/* Copyright (c) 2019 by Tim Gubner, CWI
 * Licensed under GPLv3
 */

#pragma once
#include "build.hpp"
#include "vectorized.hpp"

#include <atomic>
#include <cassert>
#include <cstring>
#include <mutex>
#include <string>

struct HashTablinho {
private:
	bucket_t *heads;

	uint32_t mod_mask;

	size_t bucket_size;

	bucket_t *next_vector;
	size_t next_stride;

	uint32_t *hash_vector;
	size_t hash_stride;

	int32_t *key_vector;
	size_t key_stride;

	std::atomic<int64_t> num_buckets;
	size_t max_buckets;

	char *value_space;

	static void _Insert(bucket_t *CPU_R buckets, bucket_t *CPU_R heads, uint32_t *CPU_R hash, uint32_t mod_mask,
	                    bucket_t *CPU_R next_vector, size_t next_stride, int *CPU_R sel, int num) {
#if 1
		Vectorized::map(sel, num, [&](auto i) {
			bucket_t bucket = buckets[i];
			size_t slot = hash[i] & mod_mask;
			bucket_t head;
			{
				head = heads[slot];
				next_vector[bucket * next_stride] = head;
			}
			while (__sync_bool_compare_and_swap(&heads[slot], head, bucket))
				;
		});
#else
		Vectorized::map(sel, num, [&](auto i) {
			bucket_t bucket = buckets[i];
			size_t slot = hash[i] & mod_mask;
			bucket_t head = heads[slot];
			next_vector[bucket * next_stride] = head;
			heads[slot] = bucket;
		});
#endif
	}

	NO_INLINE static void BucketInit(bucket_t *CPU_R buckets, bucket_t *CPU_R heads, uint32_t *CPU_R hash, uint32_t mod_mask,
	                       int *CPU_R sel, int num) {
		Vectorized::map(sel, num, [&](auto i) {
			size_t slot = hash[i] & mod_mask;
			buckets[i] = heads[slot];
		});
	}

	std::mutex finalize_build_mutex;

public:
	HashTablinho(size_t bsize, size_t capacity) {
		assert(bsize > sizeof(int64_t));
		bucket_size = bsize + sizeof(bucket_t) + sizeof(uint32_t);

		assert(bucket_size % sizeof(bucket_t) == 0);
		assert(bucket_size % sizeof(uint32_t) == 0);

		// allocate value space

		heads = nullptr;
		mod_mask = 0;
		num_buckets = 1;

		max_buckets = std::max(capacity + 1, (size_t)kVecSize * 2);
		value_space = new char[bucket_size * max_buckets];

		// set pointers

		key_vector = (int32_t *)(value_space);
		hash_vector = (uint32_t *)(value_space + bsize);
		next_vector = (bucket_t *)(value_space + bsize + sizeof(uint32_t));

		key_stride = bucket_size / sizeof(int32_t);
		next_stride = bucket_size / sizeof(bucket_t);
		hash_stride = bucket_size / sizeof(uint32_t);

		assert(key_stride > 1);
		assert(next_stride > 1);
		assert(hash_stride > 1);
	}
	~HashTablinho() {
		delete[] value_space;
		if (heads) {
			delete[] heads;
		}
	}
	void Insert(int32_t *key, uint32_t *hash, int *sel, int num) {
		// Will just crash when OOM
		assert(num_buckets.load() + num <= max_buckets);
		const size_t offset = std::atomic_fetch_add(&num_buckets, (int64_t)num);

		assert(offset + num <= max_buckets);

		Vectorized::write(key_vector, key, offset, key_stride, sel, num);
		Vectorized::write(hash_vector, hash, offset, hash_stride, sel, num);
	}
	NO_INLINE void FinalizeBuild() {
		std::lock_guard<std::mutex> guard(finalize_build_mutex);

		if (heads) {
			return;
		}

		int32_t keys[kVecSize];
		uint32_t hashs[kVecSize];
		bucket_t tmp_buckets[kVecSize];

		size_t num_heads = num_buckets * 2;
		assert(num_heads > 1);

		{
			size_t power = 1;
			while (power < num_heads)
				power *= 2;
			num_heads = power;
		}

		mod_mask = num_heads - 1;
		heads = new bucket_t[num_heads];
		for (size_t i = 0; i < num_heads; i++)
			heads[i] = 0;

		size_t i = 1;
		while (i < (size_t)num_buckets) {
			const size_t num = std::min((size_t)kVecSize, num_buckets - i);

			// gather
			Vectorized::read(keys, key_vector, i, key_stride, nullptr, num);
			Vectorized::read(hashs, hash_vector, i, hash_stride, nullptr, num);

			Vectorized::map(nullptr, num, [&](auto k) { tmp_buckets[k] = k + i; });

			_Insert(tmp_buckets, heads, hashs, mod_mask, next_vector, next_stride, nullptr, num);

			i += num;
		}
	}

	struct ProbeContext {
		bool *matches;
		bucket_t *tmp_buckets;
		int *tmp_sel;
	};

	template <size_t VSIZE> struct StaticProbeContext : ProbeContext {
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

	NO_INLINE void Probe(ProbeContext &ctx, bool *matches, int32_t *keys, uint32_t *hash, int *in_sel, int in_num) {
		assert(heads);

		int *tmp_sel = ctx.tmp_sel;
		bucket_t *tmp_buckets = ctx.tmp_buckets;

		size_t n = in_num;
		int *sel = in_sel;

		// Init using heads
		BucketInit(tmp_buckets, heads, hash, mod_mask, sel, n);

		Vectorized::map_not_match_bucket_t(matches, tmp_buckets, 0, sel, n);

		n = Vectorized::select_match(tmp_sel, matches, sel, n);
		sel = tmp_sel;

		// continue for matches
		while (n) {
			// check keys
			Vectorized::check<int32_t>(matches, keys, key_vector, tmp_buckets, key_stride, sel, n);

			n = Vectorized::select_not_match(tmp_sel, matches, sel, n);

			Vectorized::gather_next(tmp_buckets, next_vector, tmp_buckets, next_stride, sel, n);

			Vectorized::map_not_match_bucket_t(matches, tmp_buckets, 0, sel, n);
			n = Vectorized::select_match(tmp_sel, matches, sel, n);
		};
	}

	// Gathers payload column, supposed to be run after Probe for fetching the
	// buckets However, the selection vector should only contain matching keys
	NO_INLINE void ProbeGather(ProbeContext &ctx, int32_t *coldata, int64_t colidx, int *sel, int n) {
		bucket_t *tmp_buckets = ctx.tmp_buckets;
		char* column = value_space + sizeof(int32_t) * colidx;
		Vectorized::gather(coldata, (int32_t*)column, tmp_buckets, next_stride, sel, n);
	}
};
