#pragma once
#include <climits>

namespace defaults {

enum {
    num_threads_per_block = 256,
    num_blocks = 32,
    num_gpu_streams = 4,  // And 2 might just be enough actually
    num_tuples_per_kernel_launch = 1 << 21,  // used for scheduling the kernel
    should_print_results = false,
    apply_compression = false,
    // should be true really, but then we'd need a turn-off switch for no
    // compression
    num_query_execution_runs = 5,
    filter_size = 256ull * 1024 * 1024 * 8,
    probe_size = 10000
};
constexpr const char kernel_variant[] = "contains_naive";

}; //defaults