#pragma once

#include "rwticket.hpp"

#include "bloomfilter/bloom_cuda.hpp"
#include "bloomfilter/util.hpp"

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
	InflightProbe* q_next = nullptr;
	InflightProbe* q_prev = nullptr;

	enum Status {
		FRESH, // No data yet, freshly created

		FILTERING, // Filtering through the bloom filtaar
		CPU_SHARE, // Filtering done, CPUs consuming results
	};

	Status status = Status::FRESH;

	FilterWrapper::cuda_probe_t *probe;
	int64_t num;
	int64_t offset;

	std::atomic<int64_t> cpu_offset;
	std::atomic<int64_t> processed;

	cudaStream_t stream;
	InflightProbe(FilterWrapper &filter, FilterWrapper::cuda_filter_t &cf, uint32_t device, 
			int64_t start, int64_t tuples_to_process) : num(tuples_to_process), offset(start) {
		cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking  & cudaEventDisableTiming);
		probe = new typename FilterWrapper::cuda_probe_t(cf, num, stream, device);

		cpu_offset = 0;
		processed = 0;
	}
	bool is_gpu_available() {
		return probe->is_done();
	}
	void wait() {
		return probe->wait();
	}

	~InflightProbe() {
		cudaStreamDestroy(stream);
		delete probe;
	}

};
#endif

struct Pipeline {
	rwticket g_queue_rwlock;

	InflightProbe* g_queue_head;
	InflightProbe* g_queue_tail;

	std::vector<HashTablinho *> hts;
	Table &table; //!< Probe relation

	std::atomic<int64_t> tuples_morsel;

#ifdef PROFILE
	std::atomic<int64_t> tuples_gpu_probe;
	std::atomic<int64_t> tuples_gpu_consume;

	std::atomic<uint64_t> num_prefilter;
	std::atomic<uint64_t> num_postfilter;
	std::atomic<uint64_t> num_prejoin;
	std::atomic<uint64_t> num_postjoin;
#endif

	params_t& params;
private:
	std::atomic<int64_t> tuples_processed;

public:
	Pipeline(std::vector<HashTablinho *>& htables, Table& t, params_t& params)
		: hts(htables), table(t), params(params) {
		tuples_processed = 0;
		tuples_morsel = 0;

#ifdef PROFILE
		tuples_gpu_probe = 0;
		tuples_gpu_consume = 0;
		num_prefilter = 0;
		num_postfilter = 0;
		num_prejoin = 0;
		num_postjoin = 0;
#endif
		ksum = 0;
		g_queue_head = nullptr;
		g_queue_tail = nullptr;
		memset(&g_queue_rwlock, 0, sizeof(g_queue_rwlock));
	}

	void reset() {
#ifdef PROFILE
		printf("TOTAL filter sel %4.2f%% -> join sel %4.2f%%\n",
			(double)num_postfilter.load() / (double)num_prefilter.load() * 100.0,
			(double)num_postjoin.load() / (double)num_prejoin.load()* 100.0);
#endif

#ifdef NOT_SURE_WHY
		assert(g_queue_head == nullptr);
		assert(g_queue_tail == nullptr);
#else
		g_queue_tail = nullptr;
		g_queue_head = nullptr;
#endif
		// memset(&g_queue_rwlock, 0, sizeof(g_queue_rwlock));

		tuples_processed = 0;
		tuples_morsel = 0;

#ifdef PROFILE
		tuples_gpu_probe = 0;
		tuples_gpu_consume = 0;
		num_prefilter = 0;
		num_postfilter = 0;
		num_prejoin = 0;
		num_postjoin = 0;
#endif
		ksum = 0;
		table.reset();
	}

	bool is_done() const {
		return tuples_processed >= table.size();
	}

	void processed_tuples(int64_t num) {
		tuples_processed += num;
	}

	int64_t get_tuples_processed() {
		return tuples_processed.load();
	}

	NO_INLINE void g_queue_add(InflightProbe* p) noexcept {
		assert(!p->q_next);
		assert(!p->q_prev);
		p->q_next = nullptr;

		rwticket_wrlock(&g_queue_rwlock);
		// printf("g_queue_add(%p)\n", p);

		p->status = InflightProbe::Status::CPU_SHARE;
		p->q_prev = g_queue_tail;

		if (!g_queue_head) {
			g_queue_head = p;
		}

		if (g_queue_tail) {
			g_queue_tail->q_next = p;
		}

		g_queue_tail = p;

		rwticket_wrunlock(&g_queue_rwlock);
	}

	NO_INLINE void g_queue_remove(InflightProbe* p) noexcept {
		rwticket_wrlock(&g_queue_rwlock);
		// printf("g_queue_remove(%p)\n", p);

		if (p->q_next) {
			p->q_next->q_prev = p->q_prev;
		}

		if (p->q_prev) {
			p->q_prev->q_next = p->q_next;
		}

		// adapt head & tails
		if (p == g_queue_tail) {
			g_queue_tail = p->q_prev;
		}

		if (p == g_queue_head) {
			g_queue_head = p->q_next;
		}

		p->status = InflightProbe::Status::FRESH;
		rwticket_wrunlock(&g_queue_rwlock);


		p->q_next = nullptr;
		p->q_prev = nullptr;
	}

	NO_INLINE InflightProbe* g_queue_get_range(int64_t& onum, int64_t& ooffset, int64_t morsel_size) noexcept {
	#if 0
		int busy = rwticket_rdtrylock(&g_queue_rwlock);

		if (busy) {
			return nullptr; // just do CPU work instead
		}
	#else
		rwticket_rdlock(&g_queue_rwlock);
	#endif
		for (InflightProbe *p = g_queue_head; p; p = p->q_next) {
			if (p->cpu_offset >= p->num) {
				continue;
			}

			int64_t off = std::atomic_fetch_add(&p->cpu_offset, morsel_size);
			if (off >= p->num) {
				continue;
			}
			// printf("g_queue_get_range(%p)\n", p);
	#ifdef DEBUG
			assert(p->status == InflightProbe::Status::CPU_SHARE);
	#endif

			size_t n = std::min(morsel_size, p->num - off);
			onum = n;
			ooffset = off;


			rwticket_rdunlock(&g_queue_rwlock);
			assert(n > 0);
			return p;
		}
		rwticket_rdunlock(&g_queue_rwlock);
		return nullptr;
	}



	std::atomic<uint64_t> ksum;
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
	int32_t payload[kVecSize * NUM_PAYLOAD];

	int64_t tuples = 0;
	int64_t tuples_morsel = 0;

	Pipeline &pipeline;
	FilterWrapper &filter;
	FilterWrapper::cuda_filter_t &cuda_filter;

	HashTablinho::StaticProbeContext<kVecSize> ctx;

	std::vector<InflightProbe*> local_inflight;

#ifdef PROFILE
	uint64_t num_prefilter = 0;
	uint64_t num_postfilter = 0;
	uint64_t num_prejoin = 0;
	uint64_t num_postjoin = 0;
#endif

	WorkerThread(int gpu_device, Pipeline &pipeline, FilterWrapper &filter,
	              FilterWrapper::cuda_filter_t &cf)
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

	NO_INLINE void do_cpu_work(Table &table, int64_t num, int64_t offset) {
		do_cpu_join(table, nullptr, nullptr, num, offset);
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

			// probe

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

				// TODO: gather some payload columns
				for (int i = 1; i < NUM_PAYLOAD; i++) {
					ht->ProbeGather(ctx, payload + (i-1)*kVecSize, i, sel, num);
				}
			}

#ifdef PROFILE
			num_postjoin += num;
#endif

			// global sum
			Vectorized::glob_sum(&ksum, keys, sel, num);

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
		auto num_threads = pipeline.params.num_threads;
		assert(num_threads > 0);
		workers.reserve(num_threads);
		for (int i = 0; i != num_threads; ++i) {
			workers.push_back(new WorkerThread(i == 0 ? 0 : 1, pipeline, filter, cf));
		}

		for (auto &worker : workers) {
			worker->join();
		}

		for (auto &worker : workers) {
			delete worker;
		}

		printf("KSum %ld tuples procssed %ld tuplesmorsel %ld\n",
			pipeline.ksum.load(),
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
	auto &table = pipeline.table;

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
			uint32_t *tkeys = (uint32_t *)table.columns[0];

			// inflight_probe.probe->result();
			if (inflight_probe->status == InflightProbe::Status::FILTERING) {
				pipeline.g_queue_add(inflight_probe);
				break;
			}

			morsel_size = pipeline.params.gpu_morsel_size;
			auto success = table.get_range(num, offset, morsel_size);
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
			inflight_probe->status = InflightProbe::Status::FILTERING;
			inflight_probe->processed = 0;
			inflight_probe->num = num;
			inflight_probe->offset = offset;
			// printf("schedule probe %p offset %ld num %ld\n", inflight_probe, offset, num);
			inflight_probe->probe->contains(&tkeys[offset], num);
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
				do_cpu_join(table, results, nullptr, num, offset + probe->offset);

				int64_t old = std::atomic_fetch_add(&probe->processed, num);
				if (old + num == probe->num) {
					// re-use or dealloc 
					printf("remove\n");
					pipeline.g_queue_remove(probe);
				}

				pipeline.processed_tuples(num);
				continue;
			}
		}

		// full CPU join
		bool success = table.get_range(num, offset, morsel_size);
		if (!success) {
			// busy waiting until the last tuple is processed
			// give others a chance
			std::this_thread::yield();
			continue;
		}
		do_cpu_work(table, num, offset);
		pipeline.processed_tuples(num);
	}

	std::atomic_fetch_add(&pipeline.ksum, ksum);
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
