#include "hash_table.hpp"

void HashTablinho::_GetNext(void** R out, void** R buckets, size_t next_offset,
		int* R sel, int num) {
	Vectorized::map(sel, num, [&] (auto i) { out[i] = BucketGetNext(buckets[i], next_offset); });
}

void HashTablinho::_Insert(void** R buckets, void** R heads, uint32_t* R hash,
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

void HashTablinho::BucketInit(void** R buckets, void** R heads, uint32_t* R hash,
		uint32_t mod_mask, int* R sel, int num) {
	Vectorized::map(sel, num, [&] (auto i) {
		size_t slot = hash[i] & mod_mask;
		buckets[i] = heads[slot];
	});
}

void HashTablinho::Insert(int32_t* key, uint32_t* hash, int* sel, int num) {
	// Will just crash when OOM
	assert(num_buckets.load() + num <= max_buckets);
	const size_t offset = std::atomic_fetch_add(&num_buckets, (int64_t)num);

	assert(offset + num <= max_buckets);

	size_t o=0;
	Vectorized::map(sel, num, [&] (auto i) {
		char* bucket = &value_space[bucket_size * (offset + o)];

		memcpy(bucket, &key[i], sizeof(int32_t));
		memcpy(BucketPtrOfHash(bucket, hash_offset), &hash[i], sizeof(uint32_t));

		o++;
	});
}

void HashTablinho::FinalizeBuild() {
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