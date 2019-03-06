#include "hash_table.hpp"
#include "table.hpp"
#include <vector>

struct Pipeline {
private:
	uint32_t hashs[Vectorized::kVecSize];
	HashTablinho::StaticProbeContext<Vectorized::kVecSize> ctx;

	void build_vec(HashTablinho* ht, int32_t* keys, uint32_t* hashs,
			int* sel, int num) {
		Vectorized::map_hash(hashs, keys, sel, num);
		ht->Insert(keys, hashs, sel, num);
	}

	void probe_vec(HashTablinho* ht, int32_t* keys, uint32_t* hashs,
			int* sel, int num, HashTablinho::ProbeContext& ctx) {
		bool matches[num];

		Vectorized::map_hash(hashs, keys, sel, num);
		ht->Probe(ctx, matches, keys, hashs, sel, num);
	}
public:
	void build(HashTablinho* ht, Table& t, int64_t morsel_size, size_t vsize) {
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
		t.chunk([&] (void* base_addr, auto offset, auto num) {
			int32_t* tkeys = (int32_t*)base_addr;

			Vectorized::chunk(offset, num, [&] (auto offset, auto num) {
				probe_vec(ht, &tkeys[offset], hashs, nullptr, num, ctx);
			}, vsize);
		}, [&] () { });
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