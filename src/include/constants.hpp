#pragma once
#include <climits>

constexpr const size_t NUMBER_OF_STREAMS = 4;
constexpr const size_t kVecSize          = 1024;

#define NUM_PAYLOAD 32

namespace defaults {

enum {
	gpu_morsel_size = 16ull * 1024ull * 1024ull,
	cpu_morsel_size = 16 * 1024,

	num_threads_per_block = 256,
	num_blocks = 32,
	num_gpu_streams = 4,                    // at least 4
	should_print_results = false,
	num_query_execution_runs = 5,
	filter_size = 64ull * 1024ull * 1024ull * 8ull, // 64MiB
	build_size = 1024ull * 1024ull * 4ull,
	probe_size = 1024ull * 1024ull * 1024ull,
	num_repetitions = 3,
	gpu = true,
	cpu_bloomfilter = 0,
	selectivity = 1,
	num_columns = 1,
	only_generate = false,
	slowdown = 0,
	num_warmup = 2,
};
constexpr const char   kernel_variant[]  = "contains_baseline";
}; // namespace defaults
