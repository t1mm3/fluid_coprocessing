#pragma once

// #define GPU_DEBUG

#include "bloomfilter/bloom_cuda.hpp"
#include "bloomfilter/util.hpp"

#include "hash_table.hpp"
#include "query.hpp"
#include "vectorized.hpp"
#include "constants.hpp"
#include "pipeline.hpp"
#include "profiling.hpp"
#include "inflight_probe.hpp"

#include <unistd.h>
#include <atomic>
#include <thread>
#include <vector>

struct WorkerThread;

static void ExecuteWorkerThread(WorkerThread *ptr);

struct WorkerThread {
	std::thread* thread;
	int device = -1;
	uint32_t hashs[kVecSize];
	bool matches[kVecSize];
	int sel1[kVecSize];
	int sel2[kVecSize];
	int sel3[kVecSize];

	uint64_t ksum = 0;
	uint64_t psum = 0;

	int32_t payload[kVecSize * NUM_PAYLOAD];

	int64_t tuples = 0;
	int64_t tuples_morsel = 0;

	Pipeline &pipeline;
	FilterWrapper &filter;
	FilterWrapper::cuda_filter_t &cuda_filter;

	HashTablinho::StaticProbeContext<kVecSize> ctx;

	std::vector<InflightProbe*> local_inflight;

	Profiling::Time prof_aggr_cpu;
	Profiling::Time prof_aggr_gpu;
	Profiling::Time prof_aggr_gpu_cpu_join;

#ifdef PROFILE
	uint64_t num_prefilter = 0;
	uint64_t num_postfilter = 0;
	uint64_t num_prejoin = 0;
	uint64_t num_postjoin = 0;
#endif

	// temporary vectors for slowdown
	int64_t tmp1[kVecSize];
	int64_t tmp2[kVecSize];
	int64_t tmp3[kVecSize];



	WorkerThread(int gpu_device, Pipeline &pipeline, FilterWrapper &filter,
	              FilterWrapper::cuda_filter_t &cf, ProfilePrinter &profile_printer)
	    : pipeline(pipeline), device(gpu_device), filter(filter), cuda_filter(cf) {
	    thread = new std::thread(ExecuteWorkerThread, this);
	}

	void join() {
		assert(thread->joinable());
		thread->join();
	}

	~WorkerThread() {
		for (auto &inflight_probe : local_inflight) {
			delete inflight_probe;
			inflight_probe = nullptr;
		}
		delete thread;
	}

	NO_INLINE void execute_pipeline();

	NO_INLINE void do_cpu_work(Table &table, int64_t mnum, int64_t moffset) {
		Profiling::Scope profile(prof_aggr_cpu);

		do_cpu_join(table, nullptr, nullptr, mnum, moffset);
	}


	NO_INLINE void do_cpu_join(Table &table, uint32_t *bf_results, int *sel, int64_t mnum, int64_t moffset) {
		if (sel) {
			assert(mnum <= kVecSize);
		}
		tuples_morsel += mnum;

		size_t num_tuples = mnum;

		//std::cout << "morsel moffset " << moffset << " mnum " << mnum << std::endl;

		Vectorized::chunk(moffset, mnum, [&](auto offset, auto num) {
			//std::cout << "chunk offset " << offset << " num " << num << std::endl;
			int32_t *tkeys = (int32_t*)table.columns[0];
			auto keys = &tkeys[offset];

			assert(offset >= moffset);

			size_t old_num = num;

			if (bf_results) {
				//printf("cpu morsel bf_results %p offset %ld num %ld moffset %ld mnum %ld\n",
				//	bf_results, offset, num, moffset, mnum);
				const auto n = num;
				assert(pipeline.params.gpu_morsel_size % 8 == 0);
#ifdef PROFILE
				num_prefilter += n;
#endif
				num = Vectorized::select_match_bit(true, sel2,
					(uint8_t*)bf_results + (offset - moffset)/8, n);
				assert(num <= n);

#ifdef PROFILE
				num_postfilter += num;
#endif
				if (!num) {
					return; // nothing to do with this stride
				}


				sel = &sel2[0];
			} else {
				sel = nullptr;
			}

			// CPU bloom filter, if nothing else filters (creates a 'sel')
			if (!sel && pipeline.params.cpu_bloomfilter) {
#ifdef PROFILE
				num_prefilter += num;
#endif
				num = filter.contains_sel(&sel3[0], keys, sel, num);
				sel = &sel3[0];
#ifdef PROFILE
				num_postfilter += num;
#endif
			}

			// Other pipeline stuff
			if (pipeline.params.slowdown > 0) {
				Vectorized::expensive_op(pipeline.params.slowdown,
					tmp1, tmp2, tmp3, sel, num);
			}

			// Hash probe
			Vectorized::map_hash(hashs, keys, sel, num);
#ifdef PROFILE
			num_prejoin += num;
#endif
			for (auto ht : pipeline.hts) {
				ht->Probe(ctx, matches, keys, hashs, sel, num);
				num = Vectorized::select_match(sel1, matches, sel, num);
				if (!num) {
					return; // nothing to do with this stride
				}

				sel = &sel1[0];

				// gather some payload columns
				for (int i = 1; i < NUM_PAYLOAD; i++) {
					ht->ProbeGather(ctx, payload + (i-1)*kVecSize, i, sel, num);
				}
			}

#ifdef PROFILE
			num_postjoin += num;
#endif

			// global sum
			Vectorized::glob_sum(&ksum, keys, sel, num);

			for (int i = 1; i < NUM_PAYLOAD; i++) {
				Vectorized::glob_sum(&psum, payload + (i-1)*kVecSize, sel, num);
			}

			tuples += num;
		});
	}
};

void ExecuteWorkerThread(WorkerThread *ptr) {
	ptr->execute_pipeline();
}

class TaskManager {
public:

	void execute_query(Pipeline &pipeline,  FilterWrapper &filter,  FilterWrapper::cuda_filter_t &cf, ProfilePrinter &profile_printer) {
		std::vector<WorkerThread*> workers;
		auto num_threads = pipeline.params.num_threads;
		assert(num_threads > 0);
		workers.reserve(num_threads);
		for (int i = 0; i != num_threads; ++i) {
			workers.push_back(new WorkerThread(i == 0 ? 0 : 1, pipeline, filter, cf, profile_printer));
		}

		for (auto &worker : workers) {
			worker->join();
		}

		for (auto &worker : workers) {
			delete worker;
		}

		printf("KSum %ld PSum %ld tuples procssed %ld tuplesmorsel %ld\n",
			pipeline.ksum.load(), pipeline.psum.load(),
			pipeline.get_tuples_processed(), pipeline.tuples_morsel.load());

#ifdef PROFILE
		printf("gpu probe %ld gpu consumed %ld\n",
			pipeline.tuples_gpu_probe.load(),
			pipeline.tuples_gpu_consume.load());
#endif
	}
};

void WorkerThread::execute_pipeline() {
	int64_t morsel_size;
	uint64_t iteration = 0;

#ifdef HAVE_CUDA
	if (pipeline.params.gpu && device == 0) {
		cudaSetDevice(device);
		int64_t offset = 0;
		const int64_t tuples = pipeline.params.gpu_morsel_size;
		for (int i = 0; i < NUMBER_OF_STREAMS; i++) {
			// create probes
			local_inflight.push_back(new InflightProbe(filter, cuda_filter, device, offset, tuples));
			offset += tuples;
		}
	}
#endif

	while (!pipeline.is_done()) {
		int64_t num = 0;
		int64_t offset = 0;

		iteration++;

#ifdef HAVE_CUDA
		// keep GPU(s) busy
		for (auto &inflight_probe : local_inflight) {
			if (inflight_probe->status == InflightProbe::Status::CPU_SHARE) {
				continue;
			}
			if (!inflight_probe->is_gpu_available()) {
				continue;
			}
			// get the keys to probe
			uint32_t *tkeys = (uint32_t *)pipeline.table.columns[0];

			// inflight_probe.probe->result();
			if (inflight_probe->status == InflightProbe::Status::FILTERING) {
				prof_aggr_gpu.aggregate(Profiling::Time::diff(Profiling::stop(false),
					inflight_probe->prof_start));
#ifdef GPU_DEBUG
				printf("%d: cpu share %p\n",
					std::this_thread::get_id(), inflight_probe);
#endif
				pipeline.g_queue_add(inflight_probe);
				break;
			}

			morsel_size = pipeline.params.gpu_morsel_size;
			auto success = pipeline.table.get_range(num, offset, morsel_size);
			if (!success) {
				break;
			}

			// issue a new GPU BF probe
			assert(num <= pipeline.params.gpu_morsel_size);

			if (inflight_probe->status != InflightProbe::Status::FRESH) {
				assert(inflight_probe->processed >= inflight_probe->num);
			}

#ifdef PROFILE
			std::atomic_fetch_add(&pipeline.tuples_gpu_probe, num);
#endif

			// printf("probe schedule(%p)\n", inflight_probe);
			inflight_probe->reset(offset, num);
			inflight_probe->status = InflightProbe::Status::FILTERING;
			inflight_probe->prof_start = Profiling::start(true);
			inflight_probe->probe->contains(&tkeys[offset], num);
#ifdef GPU_DEBUG
			printf("%d: schedule probe %p offset %ld num %ld\n",
				std::this_thread::get_id(), inflight_probe, offset, num);
#endif
		}
#endif

		// do CPU work
		morsel_size = pipeline.params.cpu_morsel_size;

		{ // preferably do CPU join on GPU filtered data
			InflightProbe* probe = pipeline.g_queue_get_range(num, offset, morsel_size);

			if (probe) {
#ifdef PROFILE
				std::atomic_fetch_add(&pipeline.tuples_gpu_consume, num);
#endif
				uint32_t* results = probe->probe->get_results();
				// printf("cpu morsel probe %p offset %ld num %ld bf_results %p\n", probe, offset, num, results);
				assert(results != nullptr);

				{
					Profiling::Scope prof(prof_aggr_gpu_cpu_join);
					do_cpu_join(pipeline.table, results, nullptr, num, offset + probe->offset);
				}

				int64_t old = std::atomic_fetch_add(&probe->processed, num);
				assert(old + num <= probe->num);
#if 1
				if (old + num == probe->num) {
					// re-use or dealloc 
#ifdef GPU_DEBUG
					printf("%d: remove %p\n",
						std::this_thread::get_id(), probe);
#endif
					pipeline.g_queue_remove(probe);
				}
#endif
				pipeline.processed_tuples(num);
				continue;
			}
		}

		// full CPU join
		bool success = pipeline.table.get_range(num, offset, morsel_size);
		if (!success) {
			// busy waiting until the last tuple is processed
			// give others a chance
			std::this_thread::yield();
			usleep(8*1024);
			continue;
		}
		do_cpu_work(pipeline.table, num, offset);
		pipeline.processed_tuples(num);
	}

	pipeline.prof_aggr_cpu.atomic_aggregate(prof_aggr_cpu);
	pipeline.prof_aggr_gpu.atomic_aggregate(prof_aggr_gpu);
	pipeline.prof_aggr_gpu_cpu_join.atomic_aggregate(prof_aggr_gpu_cpu_join);
	std::atomic_fetch_add(&pipeline.ksum, ksum);
	std::atomic_fetch_add(&pipeline.psum, psum);
	std::atomic_fetch_add(&pipeline.tuples_morsel, tuples_morsel);
#ifdef PROFILE
	std::atomic_fetch_add(&pipeline.num_prefilter, num_prefilter);
	std::atomic_fetch_add(&pipeline.num_postfilter, num_postfilter);
	std::atomic_fetch_add(&pipeline.num_prejoin, num_prejoin);
	std::atomic_fetch_add(&pipeline.num_postjoin, num_postjoin);
#endif

#ifdef PROFILE
	printf("THREAD filter sel %4.2f%% -> join sel %4.2f%% \n",
		(double)num_postfilter / (double)num_prefilter * 100.0,
		(double)num_postjoin / (double)num_prejoin * 100.0);
#endif
}
