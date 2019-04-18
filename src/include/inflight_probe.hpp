#pragma once

#include "bloomfilter/bloom_cuda.hpp"
#include "bloomfilter/util.hpp"
#include "profiling.hpp"
#include <atomic>
#include <amsfilter/amsfilter_lite.hpp>
#include <amsfilter/internal/blocked_bloomfilter_template.hpp>

struct InflightProbe {
	InflightProbe* q_next = nullptr;
	InflightProbe* q_prev = nullptr;

	enum Status {
		FRESH, // No data yet, freshly created

		FILTERING, // Filtering through the bloom filtaar
		CPU_SHARE, // Filtering done, CPUs consuming results

		DONE, // Done ready to be reused
	};

	Status status = Status::FRESH;

	amsfilter::cuda::ProbeLite *probe;
	int64_t num;
	int64_t offset;

	std::atomic<int64_t> cpu_offset;
	std::atomic<int64_t> processed;

	Profiling::Time prof_start;
	FilterWrapper &filter;

	//cudaStream_t stream;
	InflightProbe(FilterWrapper &filter_wrapper, uint32_t device, 
			int64_t start, int64_t tuples_to_process, bool in_gpu_keys) : num(tuples_to_process), offset(start), filter(filter_wrapper) {

		probe = new typename amsfilter::cuda::ProbeLite(filter.bloom_filter.batch_probe_cuda(tuples_to_process, device));

		cpu_offset = 0;
		processed = 0;
	}
	bool is_gpu_available() {
		if (status == Status::FILTERING) {
			return probe->is_done();
		}
		return true;
	}

	void contains(uint32_t *keys, int64_t key_cnt, int64_t offset, bool in_gpu_keys) {
		auto keys_ptr = &keys[0];
		if(in_gpu_keys) {
			keys_ptr = (uint32_t*)&filter.device_keys[offset];
		}
		probe->operator()(keys_ptr, key_cnt, in_gpu_keys);
		//assert(!probe->is_done());
	}

	void wait() {
		return probe->wait();
	}

	void reset(int64_t noffset, int64_t nnum) {
		status = Status::FRESH;
		num = nnum;
		offset = noffset;
		processed = 0;
		cpu_offset = 0;
		assert(!q_next);
		assert(!q_prev);
	}

	uint32_t* get_results() {
		assert(probe->is_done());
		return probe->get_results().begin();
	}

	~InflightProbe() {
		//cudaStreamDestroy(stream);
		delete probe;
	}

};
