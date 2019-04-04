#pragma once

// #define GPU_DEBUG
// #define GPU_SYNC
#define CPU_WORK

#ifdef DEBUG 
#define GPU_BF_CHECK_AGAINST_CPU
#endif

#include "bloomfilter/bloom_cuda.hpp"
#include "bloomfilter/util.hpp"

#include "hash_table.hpp"
#include "query.hpp"
#include "vectorized.hpp"
#include "constants.hpp"
#include "pipeline.hpp"
#include "profiling.hpp"
#include "inflight_probe.hpp"
#include "timeline.hpp"

#include <unistd.h>
#include <atomic>
#include <thread>
#include <vector>

struct WorkerThread;

static void ExecuteWorkerThread(WorkerThread *ptr);

struct TimelineEvent {
	char* name;
	int64_t offset;
	int64_t num_tuples;
	void* probe = nullptr;

	void serialize(std::ostream& f, const char* sep) const {
		f << name << sep << offset << sep << num_tuples << sep << probe;
	}
};

struct WorkerThread {
	std::thread* thread;
	int device = -1;
	uint32_t hashs[kVecSize];
	bool matches[kVecSize];
	int sel1[kVecSize];
	int sel2[kVecSize];
	int sel3[kVecSize];
	int id_sel[kVecSize];
	uint8_t tmp8[kVecSize];

	uint64_t ksum = 0;
	uint64_t psum = 0;

	int32_t *payload;

	int64_t tuples = 0;
	int64_t tuples_morsel = 0;

	Pipeline &pipeline;
	FilterWrapper &filter;
	FilterWrapper::cuda_filter_t &cuda_filter;
	Timeline<TimelineEvent> timeline;

	HashTablinho::StaticProbeContext<kVecSize> ctx;

	std::vector<InflightProbe*> local_inflight;

	Profiling::Time prof_aggr_cpu;
	Profiling::Time prof_join_cpu;
	Profiling::Time prof_expop_cpu;
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
	              FilterWrapper::cuda_filter_t &cf, ProfilePrinter &profile_printer,
	              Timeline<TimelineEvent>& parent_timeline, int64_t id)
	    : pipeline(pipeline), device(gpu_device), filter(filter), cuda_filter(cf),
	    	timeline(parent_timeline) {
	    payload = new int32_t[kVecSize * pipeline.params.num_payloads];
	    thread = new std::thread(ExecuteWorkerThread, this);

	    timeline.set_id(id);
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
		delete payload;
	}

	NO_INLINE void execute_pipeline();

	void _execute_pipeline() {
		Profiling::Time prof_pipeline_cycles;
		{
			Profiling::Scope __prof(prof_pipeline_cycles);	
			Vectorized::select_identity(id_sel, kVecSize);
			execute_pipeline();
		}

		pipeline.prof_pipeline_cycles.atomic_aggregate(prof_pipeline_cycles);
	}

	NO_INLINE void do_cpu_work(int64_t mnum, int64_t moffset) {
		Profiling::Scope profile(prof_aggr_cpu);

		do_cpu_join(nullptr, mnum, moffset, 0);
	}


	NO_INLINE void do_cpu_join(uint32_t *bf_results, int64_t mnum,
				int64_t moffset, int64_t probe_offset) {
		tuples_morsel += mnum;
		Table &table = pipeline.table; 

		size_t num_tuples = mnum;

		// printf("cpu_join_morsel offset=%ld num=%ld\n", moffset, mnum);
		//std::cout << "morsel moffset " << moffset << " mnum " << mnum << std::endl;
		Vectorized::chunk(moffset, mnum, [&](auto offset, auto num) {
			int* sel = nullptr;
			//std::cout << "chunk offset " << offset << " num " << num << std::endl;
			int32_t *tkeys = (int32_t*)table.columns[0];
			int32_t* keys = &tkeys[offset + probe_offset];

			assert(offset >= moffset);

			size_t old_num = num;

			if (bf_results) {
				//printf("cpu morsel bf_results %p offset %ld num %ld moffset %ld mnum %ld\n",
				//	bf_results, offset, num, moffset, mnum);
				const auto n = num;
				assert(pipeline.params.gpu_morsel_size % 8 == 0);
				assert(pipeline.params.gpu_morsel_size % 32 == 0);
				assert(offset % 32 == 0);
				assert(offset % 8 == 0);

				uint8_t* filter_base = (uint8_t*)bf_results + offset/8;

#ifdef GPU_BF_CHECK_AGAINST_CPU
				{
					assert(!sel && "not implemented");
					filter.contains_chr(&tmp8[0], keys, sel, n);

					for (int i=0; i<n; i++) {
						int w = i / 8;
						int b = i % 8;

						bool on = !!(filter_base[w] & (1 << b));
						if (on != (bool)tmp8[i]) {
							printf("@%d: gpu=%s vs cpu=%s", i,
								on ? "1" : "0", (bool)tmp8[i] ? "1" : "0");
						}
						assert(on == (bool)tmp8[i]);
					}
				}
#endif

#ifdef PROFILE
				num_prefilter += n;
#endif
				num = Vectorized::select_match_bit(true, sel2,
					filter_base, n);
				assert(num <= n);


#ifdef PROFILE
				num_postfilter += num;
#endif
				if (!num) {
					return; // nothing to do with this stride
				}


				sel = &sel2[0];
				assert(probe_offset >= 0);
			} else {
				assert(probe_offset == 0);
				sel = nullptr;
			}

			// CPU bloom filter, if nothing else filters (creates a 'sel')
			if (!sel && pipeline.params.cpu_bloomfilter) {
#ifdef PROFILE
				num_prefilter += num;
#endif
				switch (pipeline.params.cpu_bloomfilter) {
				case 1:
					num = filter.contains_sel(&sel3[0], keys, sel, num);
					sel = &sel3[0];	
					break;
				case 2:
					filter.contains_chr(&tmp8[0], keys, sel, num);
					num = Vectorized::select_match(&sel3[0], &tmp8[0], sel, num);
					sel = &sel3[0];	
					break;
				default:
					assert(false);
					break;
				}
				
#ifdef PROFILE
				num_postfilter += num;
#endif
			}

			// Other pipeline stuff
			if (pipeline.params.slowdown > 0) {
				Profiling::Scope prof(prof_expop_cpu);
				Vectorized::expensive_op(pipeline.params.slowdown,
					tmp1, tmp2, tmp3, sel ? sel : &id_sel[0], num);
			}

			{
				Profiling::Scope jprofile(prof_join_cpu);
				// Hash probe
				Vectorized::map_hash(hashs, keys, sel, num);
#ifdef PROFILE
				num_prejoin += num;
#endif

				for (auto ht : pipeline.hts) {
					ht->Probe(ctx, matches, keys, hashs, sel, num);
					num = Vectorized::select_match(sel1, matches, sel, num);
					sel = &sel1[0];
					if (!num) {
						return; // nothing to do with this stride
					}

					// gather some payload columns
					for (int i = 0; i < pipeline.params.num_payloads; i++) {
						ht->ProbeGather(ctx, payload + i*kVecSize, i+1, sel, num);
					}
				}

#ifdef PROFILE
				num_postjoin += num;
#endif

			}

			// global sum
			Vectorized::glob_sum(&ksum, keys, sel, num);

			for (int i = 0; i < pipeline.params.num_payloads; i++) {
				Vectorized::glob_sum(&psum, payload + i*kVecSize, sel, num);
			}

			tuples += num;
		});
	}
};

void ExecuteWorkerThread(WorkerThread *ptr) {
	ptr->_execute_pipeline();
}

class TaskManager {
public:

	void execute_query(Pipeline &pipeline,  FilterWrapper &filter,  FilterWrapper::cuda_filter_t &cf,
			ProfilePrinter &profile_printer, Timeline<TimelineEvent>& timeline) {
		std::vector<WorkerThread*> workers;
		auto num_threads = pipeline.params.num_threads;
		assert(num_threads > 0);
		workers.reserve(num_threads);
		for (int i = 0; i != num_threads; ++i) {
			workers.push_back(new WorkerThread(i < 4 ? 0 : 1, pipeline, filter, cf, profile_printer, timeline, i));
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

	uint32_t *tkeys = (uint32_t *)pipeline.table.columns[0];
	uint64_t iteration = 0;


#ifdef HAVE_CUDA
	if (pipeline.params.gpu && device == 0) {
		cudaSetDevice(device);
		int64_t offset = 0;
		const int64_t tuples = pipeline.params.gpu_morsel_size;
		for (int i = 0; i < 1; i++) {
			// create probes
			// printf("inflight_probe offset %ld, tuples %ld\n", offset, tuples);
			local_inflight.push_back(new InflightProbe(filter, cuda_filter, device, offset, tuples, pipeline.params.in_gpu_keys));
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

			// inflight_probe.probe->result();
			if (inflight_probe->status == InflightProbe::Status::FILTERING) {
				prof_aggr_gpu.aggregate(Profiling::Time::diff(Profiling::stop(false),
					inflight_probe->prof_start));
#ifdef GPU_DEBUG
				printf("%d: cpu share %p\n",
					std::this_thread::get_id(), inflight_probe);
#endif
				timeline.push(TimelineEvent {"FINISHPROBE", offset, num, inflight_probe});
				pipeline.g_queue_add(inflight_probe);
				break;
			}

			morsel_size = pipeline.params.gpu_morsel_size;
			auto success = pipeline.table.get_range(num, offset, morsel_size);
			if (!success) {
				break;
			}

			assert(num % 8 == 0 && "Otherwise we cannot divide num_keys by 8");

			// printf("gpu_morsel offset %ld\n", offset);

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
			timeline.push(TimelineEvent {"SCHEDPROBE", offset, num, inflight_probe});
			inflight_probe->probe->contains(&tkeys[offset], num, offset, pipeline.params.in_gpu_keys);
#ifdef GPU_SYNC
			inflight_probe->wait();
#endif

#ifdef GPU_DEBUG
			printf("%d: schedule probe %p offset %ld num %ld\n",
				std::this_thread::get_id(), inflight_probe, offset, num);
#endif
		}
#endif

		// do CPU work
		morsel_size = pipeline.params.cpu_morsel_size;
		bool lock_busy = false;

		if (pipeline.params.gpu > 0) {
			// preferably do CPU join on GPU filtered data
			InflightProbe* probe = pipeline.g_queue_get_range(num, offset, lock_busy, morsel_size);

			if (probe) {
#ifdef PROFILE
				std::atomic_fetch_add(&pipeline.tuples_gpu_consume, num);
#endif
				uint32_t* results = probe->probe->get_results();

				// printf("cpu morsel probe %p offset %ld num %ld bf_results %p\n", probe, offset, num, results);
				assert(results != nullptr);

				{
					Profiling::Scope prof(prof_aggr_gpu_cpu_join);

					timeline.push(TimelineEvent {"CPUJOIN", offset, num, probe});
					do_cpu_join(results, num, offset, probe->offset);
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

				pipeline.processed_tuples(num, true);
				continue;
			}
		}

#ifdef CPU_WORK
		morsel_size = pipeline.params.cpu_morsel_size;
		if (lock_busy) {
			// instead of busy waiting we do some useful work
			morsel_size	= kVecSize;
		}

		// full CPU join

		bool success = pipeline.table.get_range(num, offset, morsel_size);
		if (!success && lock_busy) {
			// still waiting on Queue lock
			cpu_relax();
			continue;
		}
		if (!success) {
			// busy waiting until the last tuple is processed
			// give others a chance
			std::this_thread::yield();
			usleep(8*1024);
			continue;
		}

		timeline.push(TimelineEvent {"CPUWORK", offset, num});
		do_cpu_work(num, offset);
		pipeline.processed_tuples(num, false);
#endif
	}

	pipeline.prof_aggr_cpu.atomic_aggregate(prof_aggr_cpu);
	pipeline.prof_join_cpu.atomic_aggregate(prof_join_cpu);
	pipeline.prof_expop_cpu.atomic_aggregate(prof_expop_cpu);
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

	timeline.push(TimelineEvent {"DONE", 0, 0});
}
