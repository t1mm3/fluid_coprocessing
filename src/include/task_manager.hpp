#pragma once
#include "bloomfilter/bloom_cuda.hpp"
#include "hash_table.hpp"
#include "query.hpp"
#include "vectorized.hpp"
#include <unistd.h>



#include <atomic>
#include <thread>
#include <vector>

constexpr size_t GPU_MORSEL_SIZE = 2*1024*1024;
constexpr size_t CPU_MORSEL_SIZE = 16 * 1024;
constexpr size_t NUMBER_OF_STREAMS = 4;

#ifdef HAVE_CUDA
struct InflightProbe {
	FilterWrapper::cuda_probe_t *probe;
	int64_t num{1};
	int64_t offset{1};
	bool has_done_probing = false;
	cudaStream_t stream;
	InflightProbe(FilterWrapper &filter, FilterWrapper::cuda_filter_t &cf, uint32_t device, 
			int64_t start, int64_t tuples_to_process) : num(tuples_to_process), offset(start) {
		cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking  & cudaEventDisableTiming);
		probe = new typename FilterWrapper::cuda_probe_t(cf, num, stream, device);
	}
	bool is_gpu_available() {
		return probe->is_done();
	}
	void wait() {
		return probe->wait();
	}
	~InflightProbe() {
		cudaStreamDestroy(stream);
		probe = nullptr;	
	}
};
#endif

struct Pipeline {
	std::vector<HashTablinho *> hts;
	Table &table; //!< Probe relation

	volatile uint64_t ksum = 0;
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
	uint64_t ksum = 0;
	int32_t payload[kVecSize * 16];

	int64_t tuples = 0;
	int64_t tuples_morsel = 0;

	Pipeline &pipeline;
	FilterWrapper &filter;
	FilterWrapper::cuda_filter_t &cuda_filter;

	HashTablinho::StaticProbeContext<kVecSize> ctx;

	WorkerThread(int gpu_device, Pipeline &pipeline, FilterWrapper &filter,
	              FilterWrapper::cuda_filter_t &cf)
	    : pipeline(pipeline), device(gpu_device), thread(ExecuteWorkerThread, this), filter(filter), cuda_filter(cf) {
	}

	NO_INLINE void execute_pipeline();

	NO_INLINE void do_cpu_work(Table &table, int64_t num, int64_t offset) {
		int *sel = nullptr;

		// TODO: CPU bloom filter

		do_cpu_join(table, nullptr, sel, num, offset);
	}

	NO_INLINE void do_cpu_join(Table &table, uint32_t *bf_results, int *sel, int64_t mnum, int64_t moffset) {

		if (sel) {
			assert(mnum <= kVecSize);
		}
		std::cout << " doing join" << std::endl;
		tuples_morsel += mnum;

		//std::cout << "morsel moffset " << moffset << " mnum " << mnum << std::endl;

		Vectorized::chunk(moffset, mnum, [&](auto offset, auto num) {
			//std::cout << "chunk offset " << offset << " num " << num << std::endl;
			int32_t *tkeys = (int32_t*)table.columns[0];
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
				//std::cout << "num " << num << std::endl;
				num = Vectorized::select_match(sel1, matches, sel, num);
				sel = &sel1[0];
				//std::cout << "num2 " << num << std::endl;

				// TODO: gather some payload columns
				/*for (int i = 1; i < 4; i++) {
				    Vectorized::gather_next<int32_t>(payload + (i-1)*kVecSize,
				        ctx.tmp_buckets, i, sel, num);
				}*/
			}
			//Vectorized::map(sel, num, [&](auto i) {
			//	std::cout << "key " << keys[i] << " i " << i << std::endl;
			//});
			// global sum
			Vectorized::glob_sum(&ksum, keys, sel, num);
			//std::cout << " thread "<< std::this_thread::get_id() << " KSum inside " << ksum << std::endl;

			tuples += num;
		});
	}
};

void ExecuteWorkerThread(WorkerThread *ptr) {
	ptr->execute_pipeline();
}

class TaskManager {
public:

	void execute_query(Pipeline &pipeline,  FilterWrapper &filter,  FilterWrapper::cuda_filter_t &cf) {
		std::vector<WorkerThread*> workers;
		auto num_threads = 2; //2 * std::thread::hardware_concurrency();
		assert(num_threads > 0);
		for (int i = 0; i != num_threads; ++i) {
			workers.push_back(new WorkerThread(i == 0 ? 0 : 1, pipeline, filter, cf));
		}
		for (auto &worker : workers) {
			worker->thread.join();
			delete worker;
		}

		std::cout << "KSum " << pipeline.ksum << std::endl;
	}
};

void WorkerThread::execute_pipeline() {
	int64_t morsel_size;
	auto &table = pipeline.table;

#ifdef HAVE_CUDA
	std::vector<InflightProbe*> inflight_probes;

	if (device >= 0) {
		cudaSetDevice(device);
		int64_t offset = 0;
		int64_t tuples = GPU_MORSEL_SIZE / NUMBER_OF_STREAMS;
		for (int i = 0; i < NUMBER_OF_STREAMS; i++) {
			// create probes
			inflight_probes.push_back(new InflightProbe(filter, cuda_filter, device, offset, tuples));
			offset+= GPU_MORSEL_SIZE / NUMBER_OF_STREAMS;
		}
	}
#endif

	while (1) {
		int64_t num = 0;
		int64_t offset = 0;
		size_t finished_probes = 0;

#ifdef HAVE_CUDA
		for (auto &inflight_probe : inflight_probes) {
			if (inflight_probe->is_gpu_available()) {
				// get the keys to probe
				uint32_t* tkeys = (uint32_t *)table.columns[0];

				if (inflight_probe->has_done_probing) {
					std::cout << "hasdone " << std::endl;
					uint32_t* results = inflight_probe->probe->get_results();
					assert(results != nullptr);
					do_cpu_join(table, results, nullptr, inflight_probe->num, inflight_probe->offset);
				}

				finished_probes++;
				morsel_size = GPU_MORSEL_SIZE;
				auto success = table.get_range(num, offset, morsel_size);
				if (!success) {
					finished_probes = 0;
					break;
				}
				std::cout << "probing " << std::endl;
				inflight_probe->probe->contains(&tkeys[offset], GPU_MORSEL_SIZE);
				inflight_probe->has_done_probing = true;
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
	for (auto &probe : inflight_probes) {
		std::cout << "getting results " << std::endl;
		probe->wait();
		uint32_t* results = probe->probe->get_results();
		do_cpu_join(table, results, nullptr, probe->num, probe->offset);

	}
#endif

	//std::cout << " thread "<< std::this_thread::get_id() << " tuples " << tuples << std::endl;
//std::cout << " thread "<< std::this_thread::get_id() << " morseltuples " << tuples_morsel << std::endl;
//std::cout << " thread "<< std::this_thread::get_id() << " ksum " << ksum << std::endl;
	__sync_fetch_and_add(&pipeline.ksum, ksum);
}
