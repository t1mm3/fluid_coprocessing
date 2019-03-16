#pragma once
#include <climits>

constexpr const size_t GPU_MORSEL_SIZE   = 2 * 1024;
constexpr const size_t CPU_MORSEL_SIZE   = 16 * 1024;
constexpr const size_t NUMBER_OF_STREAMS = 4;
constexpr const size_t TABLE_SIZE        = 100000;
constexpr const size_t kVecSize          = 1024;

namespace defaults {

enum {
	num_threads_per_block = 256,
	num_blocks = 32,
	num_gpu_streams = 4,                    // at least 4
	should_print_results = false,
	num_query_execution_runs = 5,
	filter_size = 64ull * 1024 * 1024 * 8, // 64MiB
	build_size = 1024,
	probe_size = 64 * 1024,
	gpu_morsel_size = 16 * 1024,
	cpu_morsel_size = 16 * 1024,
	selectivity = 1
};
constexpr const char   kernel_variant[]  = "contains_baseline";
}; // namespace defaults
