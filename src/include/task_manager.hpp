#pragma once
#include "bloomfilter/bloom_cuda.hpp"
#include "bloomfilter/bloom_cuda_filter.hpp"
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
	// The cuda bloom filter probe logic
	cuda_probe_t probe;
	int64_t num;
	int64_t offset;
	// the stream to transfer data
	cudaStream_t stream;

	InflightProbe(const cuda_filter &cf) {
		cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
		probe = probe(cf, batch_size_gpu, stream);
	}

	bool is_gpu_available() {
		return probe.is_done();
	}

	~InflightProbe() {
		cudaStreamDestroy(stream);
	}
};
#endif

struct Pipeline {
	std::vector<HashTablinho *> hts;
	Table &table; //!< Probe relation
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

	HashTablinho::StaticProbeContext<kVecSize> ctx;

	WorkerThread(int gpu_device, const Pipeline &pipeline, const FilterWrapper &filter)
	    : pipeline(pipeline), device(gpu_device), thread(ExecuteWorkerThread, this), filter(filter) {
	}

	NO_INLINE void execute_pipeline();

	NO_INLINE void do_cpu_work(Table &table, int64_t num, int64_t offset) {
		int *sel = nullptr;

		// TODO: CPU bloom filter

		do_cpu_join(table, nullptr, sel, num, offset);
	}

	NO_INLINE void do_cpu_join(Table &table, int32_t *bf_results, int *sel, int64_t num, int64_t offset) {

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
	}

#ifdef HAVE_CUDA
	NO_INLINE void do_gpu_work(probe, offset, num) {
		probe.contains(offset, num);
	}
#endif
};

void ExecuteWorkerThread(WorkerThread *ptr) {
	ptr->execute_pipeline();
}

class TaskManager {
public:
	void execute_query(const Pipeline &pipeline) {
	}

	void execute_query(const Pipeline &pipeline, const FilterWrapper &filter) {
		std::vector<WorkerThread> workers;

		for (int i = 0; i != std::thread::hardware_concurrency(); ++i) {
			workers.emplace_back(WorkerThread(true, pipeline, filter));
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

	/*#ifdef HAVE_CUDA
	    std::vector<InflightProbe> inflight_probes;

	    if (device > 0) {
	        //instantiate CUDA Filter
	        auto cuda_filter =  cf(filter, &filter_data[0], filter_data.size());
	        for (int i = 0; i < number_of_streams; i++) {
	            //create probes
	            probes.emplace_back(InflightProbe(cf));
	        }
	    }
	#endif

	    while(1) {
	        int64_t num = 0;
	        int64_t offset = 0;
	        size_t finished_probes = 0;

	#ifdef HAVE_CUDA
	        for(auto &inflight_probe : inflight_probes) {
	            if(inflight_probe.is_gpu_available()) {
	                //get the keys to probe
	                int32_t* tkeys = (int32_t*)table.columns[0];
	                auto keys = &tkeys[offset];

	                inflight_probe.probe.contains(&keys[0], GPU_MORSEL_SIZE);
	                auto results = probe.probe.result();
	                do_cpu_join(table, results, probe.num, probe.offset);
	                finished_probes++;
	                morsel_size = GPU_MORSEL_SIZE;
	                auto success = table.get_range(num, offset, morsel_size);
	                if (!success)
	                    break;
	                do_gpu_work(probe.probe, offset, num);
	            }
	        }
	#endif
	        if (finished_probes == 0) {
	            morsel_size = CPU_MORSEL_SIZE;
	            auto success = table.get_range(num, offset, morsel_size);
	            if (!success)
	                break;
	            do_cpu_work(table, num, offset);
	        }
	    }
	#ifdef HAVE_CUDA
	    for(auto &probe : inflight_probes){
	        probe.wait();
	        auto results = probe.probe.result();
	        do_cpu_join(table, results, probe.num, probe.offset);
	    }
	#endif*/
}
