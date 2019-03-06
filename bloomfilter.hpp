#define R __restrict

#include <malloc.h>
#include <stdint.h>

static uint64_t rdtsc()
{
	unsigned hi, lo;
	__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
	return ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );
}


#define NO_INLINE __attribute__((noinline))

struct Vectorized {
	static constexpr size_t kVecSize = 1024;

	template<typename T>static void
	map(int* sel, int num, T&& fun) {
		if (sel) {
			for (int i=0; i<num; i++) {
				fun(sel[i]);
			}
		} else {
			for (int i=0; i<num; i++) {
				fun(i);
			}
		}
	}

	template<typename T>static int
	select(int* osel, int* sel, int num, T&& fun) {
		int res = 0;
		if (sel) {
			for (int i=0; i<num; i++) {
				if (fun(sel[i])) {
					osel[res++] = sel[i];
				}
			}
		} else {
			for (int i=0; i<num; i++) {
				if (fun(i)) {
					osel[res++] = i;
				}
			}
		}

		return res;
	}

	static void NO_INLINE map_not_match_void(bool* R out, void** R a,
			void* R b, int* R sel, int num) {
		map(sel, num, [&] (auto i) { out[i] = a[i] != b; });
	}

	static int NO_INLINE select_match(int* R osel, bool* R b, int* R sel,
			int num) {
		return select(osel, sel, num, [&] (auto i) { return b[i]; });
	}

	static int NO_INLINE select_not_match(int* R osel, bool* R b, int* R sel,
			int num) {
		return select(osel, sel, num, [&] (auto i) { return !b[i]; });
	}

	inline static uint32_t hash32(uint32_t a) {
		return a * 2654435761;
	}

	static void NO_INLINE map_hash(uint32_t* R out, int32_t* R a, int* R sel,
			int num) {
		map(sel, num, [&] (auto i) { out[i] = hash32((uint32_t)(a[i])); });
	}

	template<typename T>
	static int NO_INLINE check(bool* R match, T* R keys, T* R table, size_t* R idx,
			size_t stride, int* R sel, int num) {
		if (stride > 1) {
			map(sel, num, [&] (auto i) { match[i] = table[idx[i] * stride] == keys[i]; });
		} else {
			map(sel, num, [&] (auto i) { match[i] = table[idx[i] * 1] == keys[i]; });
		}
	}

	template<typename T>
	static int NO_INLINE check_ptr(bool* R match, T* R keys, T** R ptrs, int* R sel, int num) {
		map(sel, num, [&] (auto i) { match[i] = (*ptrs[i]) == keys[i]; });
	}

	template<typename T>
	static void chunk(size_t offset, size_t size, T&& fun, int vsize = kVecSize) {
		const size_t end = size + offset;
		size_t i = offset;
		while (i < end) {
			size_t num = std::min(vsize, end-i);
			fun(i, num);
		}
	}
};

#include <atomic>
#include <cassert>
#include <mutex>

struct HashTablinho {
	void** heads;

	uint32_t mod_mask;

	size_t bucket_size;
	size_t next_offset;
	size_t hash_offset;

	std::atomic<size_t> num_buckets;
	size_t max_buckets;

	char* value_space;

private:
	static void** BucketPtrOfNext(void* bucket, size_t offset) {
		return (void**)((char**)bucket + offset);
	}

	static uint32_t* BucketPtrOfHash(void* bucket, size_t offset) {
		return (uint32_t**)((char**)bucket + offset);
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

		max_buckets = std::max(capacity, 1024);
		value_space = new char[bucket_size * max_buckets];
	}

	~HashTablinho() {
		delete[] value_space;
	}

	void Insert(int32_t* key, uint32_t* hash, int* sel, int num) {
		assert(num_buckets + num < max_buckets);
		const size_t offset = std::atomic_fetch_add(&num_buckets, num);

		assert(offset + num < max_buckets);

		size_t o=0;
		Vectorized::map(sel, num, [&] (auto i) {
			char* bucket = &value_space[bucket_size * (offset + o)];

			memcpy(bucket, key[i], sizeof(int32_t));
			memcpy(BucketPtrOfHash(bucket, hash_offset), hash[i], sizeof(uint32_t));

			o++;
		});
	}

	static constexpr size_t kVecSize = Vectorized::kVecSize;

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
		while (i < num_buckets) {
			size_t num = std::min(kVecSize, num_buckets-i);

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
				check_ptr<int32_t>(match, keys, (int32_t**)buckets, sel, n);
		});
	}
};

#include <atomic>

struct Table {
private:
	int64_t capacity;
	std::atomic<int64_t> start; //! Current start offset
	std::atomic<int64_t> done; //!< Current finished offset (after morsel completion)

	void *base;

public:
	int64_t size() const { return capacity; }

	Table(size_t cap) {
		start = 0;
		done = 0;
		capacity = cap;
	}

	void reset() {
		done = 0;
		start = 0;
	}

private:
	/* "Consumes" a range of the table
		\param ostart Offset of range
		\param onum Number of tuples in range
		\param num Preferred size of range
		\returns True, if successful (and valid range); False, otherwise
	*/
	bool get_range(int64_t& onum, int64_t& ostart, int64_t num) {
		if (start > capacity) {
			return false;
		}

		ostart = std::atomic_fetch_add(&start, num);
		if (ostart >= capacity) {
			return false;
		}

		const int64_t todo = capacity - ostart;
		assert(todo > 0);

		onum = std::min(todo, num);
		return onum > 0;
	}

public:
	template<typename T, typename S>
	void chunk(T&& morsel, S&& finished, size_t morsel_size = 16*1024) {
		while (1) {
			int64_t num = 0;
			int64_t offset = 0;
			bool has_range = get_range(num, offset, morsel_size);

			if (!has_range) {
				break;
			}

			assert(num <= morsel_size);

			morsel(base, offset, num);

			done += num;
			if (done >= capacity) {
				finished();
			}
		}
	}
};

#include <vector>

struct Pipeline {
private:
	void build_vec(HashTablinho* ht, int32_t* keys, uint32_t* hashs,
			int* sel, int num) {
		Vectorized::map_hash(hashs, keys, sel, num);
		ht->Insert(keys, hashs, sel, num);
	}

	void probe_vec(HashTablinho* ht, int32_t* keys, uint32_t* hashs,
			int* sel, int num) {
		Vectorized::map_hash(hashs, keys, sel, num);
		// ht->Insert(keys, hashs, sel, num);
	}
public:
	void build(HashTablinho* ht, Table& t, int64_t morsel_size, size_t vsize) {
		uint32_t hashs[vsize];

		t.chunk([&] (void* base_addr, auto offset, auto num) {
			int32_t* tkeys = (int32_t*)base_addr;
			Vectorized::chunk(offset, num, [&] (auto offset, auto num) {
				build_vec(ht, &tkeys[offset], hashs, nullptr, num);
			}, vsize);
		}, [&] () {
			// finished
			ht->FinalizeBuild();
		});
	}

	void probe(HashTablinho* ht, Table& t, int64_t morsel_size, size_t vsize) {
		uint32_t hashs[vsize];

		t.chunk([&] (void* base_addr, auto offset, auto num) {
			int32_t* tkeys = (int32_t*)base_addr;

			Vectorized::chunk(offset, num, [&] (auto offset, auto num) {
				probe_vec(ht, &tkeys[offset], hashs, nullptr, num);
			}, vsize);
		}, [&] () {
			// finished
			ht->FinalizeBuild();
		});
	}
};

#include <vector>
#include <thread>

struct Scheduler {
private:
	std::vector<std::thread> workers;

	std::atomic<bool> done;

	template<bool GPU>
	void worker() {
		while (!done) {
			
		}
	}

	void NO_INLINE cpu() { return worker<false>(); }
	void NO_INLINE gpu() { return worker<true>(); }
public:
	~Scheduler() {
		for (auto& w : workers) {
			w.join();
		}
	}

	Scheduler() {
		std::thread t(&Scheduler::cpu, this);
	}
};