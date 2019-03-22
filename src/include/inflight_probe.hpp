#pragma once

#include "bloomfilter/bloom_cuda.hpp"
#include "bloomfilter/util.hpp"
#include "profiling.hpp"
#include <atomic>

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

	FilterWrapper::cuda_probe_t *probe;
	int64_t num;
	int64_t offset;

	std::atomic<int64_t> cpu_offset;
	std::atomic<int64_t> processed;

	Profiling::Time prof_start;

	cudaStream_t stream;

	bool emulate;

	FilterWrapper &filter;

	InflightProbe(FilterWrapper &filter, FilterWrapper::cuda_filter_t &cf, uint32_t device, 
			int64_t start, int64_t tuples_to_process, bool emulate)
			: num(tuples_to_process), offset(start), emulate(emulate), filter(filter) {
		cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking  & cudaEventDisableTiming);
		probe = new typename FilterWrapper::cuda_probe_t(cf, num, stream, device);

		cpu_offset = 0;
		processed = 0;
	}
	bool is_gpu_available() {
		if (emulate) {
			return true;
		}
		return probe->is_done();
	}
	void wait() {
		if (emulate) {
			return;
		}
		return probe->wait();
	}

	void contains(uint32_t* keys, int64_t num) {
		if (emulate) {
			filter.contains_bit((uint8_t*)probe->get_results(), keys, num);
			return;
		}
		probe->contains(keys, num);
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

	~InflightProbe() {
		cudaStreamDestroy(stream);
		delete probe;
	}

};
