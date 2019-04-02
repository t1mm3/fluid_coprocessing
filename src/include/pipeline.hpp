#pragma once

#include "profiling.hpp"
#include "rwticket.hpp"

#include "hash_table.hpp"
#include "inflight_probe.hpp"
#include <atomic>
#include <vector>

struct Pipeline {
	rwticket g_queue_rwlock;

	InflightProbe* g_queue_head;
	InflightProbe* g_queue_tail;

	std::vector<HashTablinho *> hts;
	Table &table; //!< Probe relation

	Profiling::Time prof_aggr_cpu;
	Profiling::Time prof_join_cpu;
	Profiling::Time prof_expop_cpu;
	Profiling::Time prof_aggr_gpu;
	Profiling::Time prof_aggr_gpu_cpu_join;
	Profiling::Time prof_pipeline_cycles;

	std::atomic<int64_t> tuples_morsel;

	std::atomic<int64_t> tuples_gpu_probe;
	std::atomic<int64_t> tuples_gpu_consume;
#ifdef PROFILE

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
		psum = 0;
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

		assert(g_queue_head == nullptr);
		assert(g_queue_tail == nullptr);
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
		psum = 0;

		prof_aggr_cpu.reset();
		prof_aggr_gpu.reset();
		prof_aggr_gpu_cpu_join.reset();
		prof_pipeline_cycles.reset();
		prof_expop_cpu.reset();
		prof_join_cpu.reset();
		table.reset();
	}

	bool is_done() const {
		return tuples_processed >= table.size();
	}

	void processed_tuples(int64_t num, bool gpu) {
		auto old = std::atomic_fetch_add(&tuples_processed, num);
#if 0
		printf("%s tuples %ld  old = %ld, %ld <= %ld (max)\n",
			gpu ? "GPU" : "cpu", num, old, old+num, table.size());
#endif
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


		p->q_next = nullptr;
		p->q_prev = nullptr;
		barrier();
		p->status = InflightProbe::Status::DONE;
		rwticket_wrunlock(&g_queue_rwlock);
	}

	NO_INLINE InflightProbe* g_queue_get_range(int64_t& onum, int64_t& ooffset,
			bool& is_busy, int64_t morsel_size) noexcept {
		is_busy = false;

		if (!g_queue_head) {
			return nullptr;
		}
#if 0
		int busy = rwticket_rdtrylock(&g_queue_rwlock);

		if (busy) {
			is_busy = true;
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

#ifdef GPU_DEBUG
			printf("%d: g_queue_get_range %p offset %ld num %ld\n",
				std::this_thread::get_id(), p, ooffset, onum);
#endif
			rwticket_rdunlock(&g_queue_rwlock);
			assert(n > 0);
			return p;
		}
		rwticket_rdunlock(&g_queue_rwlock);
		return nullptr;
	}



	std::atomic<uint64_t> ksum;
	std::atomic<uint64_t> psum;
};
