#pragma once
#include "bloomfilter/bloom_cuda.hpp"
#include "hash_table.hpp"
#include "query.hpp"
#include "vectorized.hpp"

#include <atomic>
#include <thread>
#include <vector>

constexpr size_t GPU_MORSEL_SIZE = 100 * 1024 * 1024;
constexpr size_t CPU_MORSEL_SIZE = 10 * 1024;
constexpr size_t NUMBER_OF_STREAMS = 4;

#ifdef HAVE_CUDA
struct InflightProbe {
	enum Status {
		FRESH, // No data yet, freshly created

		FILTERING, // Filtering through the bloom filtaar
		CPU_SHARE, // Filtering done, CPUs consuming results
	};

	Status status = Status::FRESH;

	FilterWrapper::cuda_probe_t *probe;
	int64_t num;
	int64_t offset;
	bool has_done_probing = false;
	cudaStream_t stream;
	InflightProbe(const FilterWrapper &filter, const FilterWrapper::cuda_filter_t &cf) {
		cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
		probe = new FilterWrapper::cuda_probe_t(cf, offset, stream);
	}
	bool is_gpu_available() {
		return true;
		// probe->is_done();
	}
	bool wait() {
		return true;
		// probe->is_done();
	}
	~InflightProbe() {
		cudaStreamDestroy(stream);
		delete probe;
	}
};
#endif


struct GlobalQueue {
	// cannot be resized @ runtime without mmap() "sorcery"
	// In any case, we assume that we are no going to have many
	// of these InflightProbe stored (4 streams per GPU).
	// Hence the size is in it's 10s rather than 100s.
	std::vector<InflightProbe*> probes;

	GlobalQueue(int64_t cap) {
		probes.reserve(cap);
		for (int64_t i=0; i<cap; i++) {
			probes.push_back(nullptr);
		}
	}

	void add(InflightProbe* mp) {
		bool found = false;
		auto num = probes.size();

		do {
			for (int64_t i=0; i<num; i++) {
				auto ptr = &probes[i];
				if (*ptr == nullptr) {
					bool success = __sync_bool_compare_and_swap(ptr, nullptr, mp);
					if (success) {
						found = true;
						break;
					}
				}
			}
		} while (!found);

		assert(found);
	}

	void remove(InflightProbe* mp) {
		bool found = false;	
		auto num = probes.size();

		for (int64_t i=0; i<num; i++) {
			auto ptr = &probes[i];
			if (*ptr == mp) {
				bool success = __sync_bool_compare_and_swap(ptr, mp, nullptr);
				if (success) {
					// we removed the element
					found = true;
					break;
				} else {
					// another thread removed the element
					assert(false);
				}
			}
		}

		assert(found);
	}

	InflightProbe* get_range(int64_t& onum, int64_t& ooffset, int64_t morsel_size) {
		todo
	}
};


struct Pipeline {
	std::vector<HashTablinho *> hts;
	Table &table; //!< Probe relation

	GlobalQueue done_probes(128);

private:
	std::atomic<int64_t> tuples_processed = 0;

public:
	bool is_done() const {
		return tuples_processed >= table.size();
	}

	void processed_tuples(int64_t num) {
		tuples_processed += num;
	}
};

struct WorkerThread;

static void ExecuteWorkerThread(WorkerThread *ptr);

struct WorkerThread {
	std::thread thread;
	int device = -1;
	uint32_t hashs[kVecSize];
	bool matches[kVecSize];
	int sel1[kVecSize];
	int sel2[kVecSize];
	int64_t ksum = 0;
	int32_t payload[kVecSize * 16];

	const Pipeline &pipeline;
	const FilterWrapper &filter;
	const FilterWrapper::cuda_filter_t &cuda_filter;

	HashTablinho::StaticProbeContext<kVecSize> ctx;

	WorkerThread(int gpu_device, const Pipeline &pipeline, const FilterWrapper &filter,
	             const FilterWrapper::cuda_filter_t &cf)
	    : pipeline(pipeline), device(gpu_device), thread(ExecuteWorkerThread, this), filter(filter), cuda_filter(cf) {
	}

	NO_INLINE void execute_pipeline();

	NO_INLINE void do_cpu_work(Table &table, int64_t num, int64_t offset) {
		int *sel = nullptr;

		// TODO: CPU bloom filter

		do_cpu_join(table, nullptr, sel, num, offset);
	}

	NO_INLINE void do_cpu_join(Table &table, uint32_t *bf_results, int *sel, int64_t num, int64_t offset) {
		const int64_t num_tuples = num;

		assert(!sel == !!bf_results);

		if (sel) {
			assert(num <= kVecSize);
		}

		Vectorized::chunk(offset, num, [&](auto offset, auto num) {
			int32_t *tkeys = (int32_t *)table.columns[0];
			auto keys = &tkeys[offset];

			if (bf_results) {
				int n = num + (8 - 1);
				n /= 8;
				n *= 8;
				// num = Vectorized::select_match_bit(sel1, bf_results + offset/8, n);
				sel = &sel2[0];
			} else {
				sel = nullptr;
			}

			// probe
			Vectorized::map_hash(hashs, keys, sel, num);

			for (auto ht : pipeline.hts) {
				ht->Probe(ctx, matches, keys, hashs, sel, num);
				num = Vectorized::select_match(sel1, matches, sel, num);
				sel = &sel1[0];

				// TODO: gather some payload columns
				/*for (int i = 1; i < 4; i++) {
				    Vectorized::gather_next<int32_t>(payload + (i-1)*kVecSize,
				        ctx.tmp_buckets, i, sel, num);
				}*/
			}

			// global sum
			Vectorized::glob_sum(&ksum, keys, sel, num);

		});

		// mark tuples as procssed
		pipeline.processed_tuples(num_tuples);
	}
};

void ExecuteWorkerThread(WorkerThread *ptr) {
	ptr->execute_pipeline();
}

class TaskManager {
public:
	void execute_query(const Pipeline &pipeline) {
	}

	void execute_query(const Pipeline &pipeline, const FilterWrapper &filter, const FilterWrapper::cuda_filter_t &cf) {
		std::vector<WorkerThread> workers;

		for (int i = 0; i != std::thread::hardware_concurrency(); ++i) {
			workers.emplace_back(WorkerThread(true, pipeline, filter, cf));
		}
		for (auto &worker : workers) {
			worker.thread.join();
		}
	}
};

/*void TaskManager::execute_query(const Pipeline& pipeline) {
    std::vector<WorkerThread> workers;

    auto hardware_threads = std::thread::hardware_concurrency();
    assert(hardware_threads > 0);

    for(int i = 0; i != hardware_threads; ++i) {
        workers.emplace_back(WorkerThread(false, pipeline));
    }
    for(auto &worker : workers) {
        worker.thread.join();
    }

}*/

void WorkerThread::execute_pipeline() {
	int64_t morsel_size;
	auto &table = pipeline.table;

	uint64_t iteration = 0;

#ifdef HAVE_CUDA
	std::vector<InflightProbe> inflight_probes;

	if (device > 0) {
		// instantiate CUDA Filter
		for (int i = 0; i < NUMBER_OF_STREAMS; i++) {
			// create probes
			inflight_probes.emplace_back(InflightProbe(filter, cuda_filter));
		}
	}
#endif

	while (!pipeline.is_done()) {
		int64_t num = 0;
		int64_t offset = 0;
		size_t finished_probes = 0;

		iteration++;

#ifdef HAVE_CUDA
		// keep GPU(s) busy
		for (auto &inflight_probe : inflight_probes) {
			if (inflight_probe.is_gpu_available()) {
				// get the keys to probe
				uint32_t *tkeys = (uint32_t *)table.columns[0];
				uint32_t *results = 0;
				// TODO results
				// inflight_probe.probe->result();
				if (inflight_probe.status == InflightProbe::Status::FILTERING) {
					inflight_probe.status = InflightProbe::Status::CPU_SHARE;
					pipeline.done_probes.add(inflight_probe);
				}

				finished_probes++;
				morsel_size = GPU_MORSEL_SIZE;
				auto success = table.get_range(num, offset, morsel_size);
				if (!success)
					break;

				// issue a new GPU BF probe
				inflight_probe.probe->contains(&tkeys[offset], morsel_size);
				inflight_probe.status = InflightProbe::Status::FILTERING;
			}
		}
#endif

		// do CPU work
		bool success = true;
		morsel_size = CPU_MORSEL_SIZE;

		{ // preferably do CPU join on GPU filtered data
			InflightProbe* probe = pipeline.done_probes.get_range(num, offset, morsel_size);

			if (probe) {
				do_cpu_join();

				if (last) {
					// re-use or dealloc 
					pipeline.done_probes.remove(probe);
					probe->status = InflightProbe::Status::FRESH;
					delete probe;
				}
				continue;
			}
		}

		// full CPU join
		success = table.get_range(num, offset, morsel_size);
		if (!success) {
			// busy waiting until the last tuple is processed
			// give others a chance
			std::this_thread::yield();
		}
		do_cpu_work(table, num, offset);
	}
}
