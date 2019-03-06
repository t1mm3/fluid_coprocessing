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

	static void NO_INLINE _GetNext(void** R out, void** R buckets, size_t next_offset,
			int* R sel, int num) {
		Vectorized::map(sel, num, [&] (auto i) { out[i] = BucketGetNext(buckets[i], next_offset); });
	}

	static void NO_INLINE _Insert(void** R buckets, void** R heads, uint32_t* R hash,
			uint32_t mod_mask, size_t next_offset, int* R sel, int num) {
		Vectorized::map(sel, num, [&] (auto i) {
			void* bucket = buckets[i];
			size_t slot = hash[i] & mod_mask;
			void* head;
			{
				head = heads[slot];
				BucketSetNext(bucket, head, next_offset);
			} while (__sync_bool_compare_and_swap(&heads[slot], head, bucket));
		});
	}

	static void NO_INLINE BucketInit(void** R buckets, void** R heads, uint32_t* R hash,
			uint32_t mod_mask, int* R sel, int num) {
		Vectorized::map(sel, num, [&] (auto i) {
			size_t slot = hash[i] & mod_mask;
			buckets[i] = heads[slot];
		});
	}

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

	void Insert(int32_t* key, uint32_t* hash, int* sel, int num) {
		// Will just crash when OOM
		assert(num_buckets.load() + num < max_buckets);
		const size_t offset = std::atomic_fetch_add(&num_buckets, (int64_t)num);

		assert(offset + num < max_buckets);

		size_t o=0;
		Vectorized::map(sel, num, [&] (auto i) {
			char* bucket = &value_space[bucket_size * (offset + o)];

			memcpy(bucket, &key[i], sizeof(int32_t));
			memcpy(BucketPtrOfHash(bucket, hash_offset), &hash[i], sizeof(uint32_t));

			o++;
		});
	}

	void FinalizeBuild() {
		std::lock_guard<std::mutex> guard(finalize_build_mutex);

		if (heads) {
			return;
		}

		size_t num_heads = num_buckets*2;
		assert(num_heads > 1);

		{
			size_t power = 1;
			while(power < num_heads) power*=2;
			num_heads = power;
		}

		mod_mask = num_heads-1;
		heads = new void*[num_heads];
		for (size_t i=0; i<num_heads; i++) heads[i] = nullptr;

		int32_t keys[kVecSize];
		uint32_t hash[kVecSize];
		void* tmp_buckets[kVecSize];

		size_t i = 0;
		while (i < (size_t)num_buckets) {
			size_t num = std::min((size_t)kVecSize, num_buckets-i);

			// gather
			Vectorized::map(nullptr, num, [&] (auto k) {
				size_t idx = k+i;
				void* bucket = &value_space[idx * bucket_size];

				hash[i] = *BucketPtrOfHash(bucket, hash_offset);
				int32_t* key = (int32_t*)bucket;
				keys[i] = *key;
			});

			_Insert(tmp_buckets, heads, hash, mod_mask, next_offset, nullptr, num);

			i+=num;
		}
	}

	struct ProbeContext {
		bool** matches;
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
};
