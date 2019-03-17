#pragma once

#include "bloomfilter/bloom_cuda.hpp"
#include "hash_table.hpp"
#include "query.hpp"
#include "vectorized.hpp"
#include "constants.hpp"

#include <unistd.h>
#include <atomic>
#include <thread>
#include <vector>

#ifdef HAVE_CUDA
struct InflightProbe {
	enum Status {
		FRESH, // No data yet, freshly created

		FILTERING, // Filtering through the bloom filtaar
		CPU_SHARE, // Filtering done, CPUs consuming results
	};

	Status status = Status::FRESH;

	FilterWrapper::cuda_probe_t *probe;
	int64_t num{1};
	int64_t offset{1};

	std::atomic<int64_t> processed;

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
					bool success = __sync_bool_compare_and_swap(ptr, (InflightProbe*)nullptr, mp);
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
				bool success = __sync_bool_compare_and_swap(ptr, mp, (InflightProbe*)nullptr);
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
		// FIXME: todo
		return nullptr;
	}
};


struct Pipeline {
	std::vector<HashTablinho *> hts;
	Table &table; //!< Probe relation

	GlobalQueue done_probes;

private:
	std::atomic<int64_t> tuples_processed;

public:
	Pipeline(std::vector<HashTablinho *>& htables, Table& t)
		: hts(htables), table(t), done_probes(128) {
		tuples_processed = 0;
	}

	bool is_done() const {
		return tuples_processed >= table.size();
	}

	void processed_tuples(int64_t num) {
		tuples_processed += num;
	}

	volatile uint64_t ksum = 0;
};

struct WorkerThread;

static void ExecuteWorkerThread(WorkerThread *ptr);

struct WorkerThread {
	std::thread* thread;
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
	    : pipeline(pipeline), device(gpu_device), filter(filter), cuda_filter(cf) {
	    thread = new std::thread(ExecuteWorkerThread, this);
	}

	~WorkerThread() {
		thread->join();
		delete thread;
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
		tuples_morsel += mnum;

		size_t num_tuples = num;

		//std::cout << "morsel moffset " << moffset << " mnum " << mnum << std::endl;

		Vectorized::chunk(moffset, mnum, [&](auto offset, auto num) {
			//std::cout << "chunk offset " << offset << " num " << num << std::endl;
			int32_t *tkeys = (int32_t*)table.columns[0];
			auto keys = &tkeys[offset];

			size_t old_num = num;

			if (bf_results) {
#if 0				
				int n = num + (8 - 1);
				n /= 8;
				n *= 8;
#else
				const auto n = num;
#endif
				num = Vectorized::select_match_bit(sel1, (uint8_t*)bf_results + offset/8, n);

				if (!num) {
					return; // nothing to do with this stride
				}

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

				if (!num) {
					return; // nothing to do with this stride
				}

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

		// mark tuples as procssed
		pipeline.processed_tuples(num_tuples);
	}
};

void ExecuteWorkerThread(WorkerThread *ptr) {
	ptr->execute_pipeline();
}

class TaskManager {
public:

	void execute_query(Pipeline &pipeline,  FilterWrapper &filter,  FilterWrapper::cuda_filter_t &cf) {
		std::vector<WorkerThread*> workers;
		auto num_threads = std::thread::hardware_concurrency();
		assert(num_threads > 0);
		for (int i = 0; i != num_threads; ++i) {
			workers.push_back(new WorkerThread(i == 0 ? 0 : 1, pipeline, filter, cf));
		}
		for (auto &worker : workers) {
			delete worker;
		}

		std::cout << "KSum " << pipeline.ksum << std::endl;
	}
};

void WorkerThread::execute_pipeline() {
	int64_t morsel_size;
	auto &table = pipeline.table;

	uint64_t iteration = 0;

#ifdef HAVE_CUDA
	std::vector<InflightProbe*> inflight_probes;

	if (device == 0) {
		cudaSetDevice(device);
		int64_t offset = 0;
		int64_t tuples = GPU_MORSEL_SIZE;
		for (int i = 0; i < NUMBER_OF_STREAMS; i++) {
			// create probes
			inflight_probes.push_back(new InflightProbe(filter, cuda_filter, device, offset, tuples));
			offset+= GPU_MORSEL_SIZE;
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
			if (inflight_probe->status == InflightProbe::SHARED) {
				continue;
			}
			if (!inflight_probe->is_gpu_available()) {
				continue;
			}
			// get the keys to probe
			uint32_t *tkeys = (uint32_t *)table.columns[0];

			// inflight_probe.probe->result();
			if (inflight_probe.status == InflightProbe::Status::FILTERING) {
				inflight_probe.status = InflightProbe::Status::CPU_SHARE;
				pipeline.done_probes.add(inflight_probe);
			}

			finished_probes++;
			morsel_size = GPU_MORSEL_SIZE;
			auto success = table.get_range(num, offset, morsel_size);
			if (!success) {
				finished_probes = 0;
				break;
			}

			// issue a new GPU BF probe
			inflight_probe.probe->contains(&tkeys[offset], num);
			inflight_probe.status = InflightProbe::Status::FILTERING;
		}
#endif

		// do CPU work
		bool success = true;
		morsel_size = CPU_MORSEL_SIZE;

		{ // preferably do CPU join on GPU filtered data
			InflightProbe* probe = pipeline.done_probes.get_range(num, offset, morsel_size);

			if (probe) {
				uint32_t* results = probe->probe->get_results();
				assert(results != nullptr);
				do_cpu_join(table, results, nullptr, num, offset + probe->offset);

				int64_t old = std::atomic_fetch_add(&probe->processed, num);
				if (old == probe->num) {
					// re-use or dealloc 
					pipeline.done_probes.remove(probe);
					probe->status = InflightProbe::Status::FRESH;
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

	__sync_fetch_and_add(&pipeline.ksum, ksum);
}
