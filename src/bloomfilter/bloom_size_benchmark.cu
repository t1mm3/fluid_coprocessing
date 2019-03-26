#include <algorithm>
#include <iomanip>
#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <string>

#include <dtl/dtl.hpp>
#include <dtl/env.hpp>
#include <dtl/hash.hpp>
#include <dtl/thread.hpp>
#include <dtl/mem.hpp>
#include <dtl/filter/blocked_bloomfilter/zoned_blocked_bloomfilter.hpp>

#include "cuda_helper.hpp"
#include "util.hpp"
#include <cub/device/device_radix_sort.cuh>
#include "cuda/blocked_bloomFilter.hpp"


//===----------------------------------------------------------------------===//
// Typedefs. (cache-sectorized blocked bloom filter)
using filter_key_t = $u32;
using hash_value_t = $u32;
using word_t       = $u32;

// The first hash function to use inside the block. Note: 0 is used for block addressing
static constexpr u32 block_hash_fn_idx = 1;

// The block type.
template<u32 word_cnt, u32 zone_cnt, u32 k, u1 early_out = false>
using bbf_block_t = dtl::multizone_block<filter_key_t, word_t, word_cnt, zone_cnt, k,
                                         dtl::hasher, hash_value_t,
                                         block_hash_fn_idx, 0, zone_cnt,
                                         early_out>;

template<u32 word_cnt, u32 zone_cnt, u32 k, dtl::block_addressing addr = dtl::block_addressing::POWER_OF_TWO, u1 early_out = false>
using bbf_t = dtl::blocked_bloomfilter_logic<filter_key_t, dtl::hasher, bbf_block_t<word_cnt, zone_cnt, k, early_out>, addr, early_out>;



//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
template<
    std::size_t word_cnt,
    std::size_t zone_cnt,
    std::size_t k
>
void benchmark(const std::size_t m,
               std::vector<filter_key_t>& to_insert,
               std::vector<filter_key_t>& to_lookup,
               std::size_t bits_to_sort,
               std::ofstream &results) {
    //===----------------------------------------------------------------------===//
    // Construct an empty filter with the given size.
    using filter_t = bbf_t<word_cnt, zone_cnt, k>;
    using word_t = typename filter_t::word_t;
    filter_t filter(m);
    
    // Allocate memory.
    dtl::mem::allocator_config alloc_config = dtl::mem::allocator_config::local();
    dtl::mem::numa_allocator<word_t> allocator(alloc_config);
    
    using filter_data_t = std::vector<word_t, dtl::mem::numa_allocator<word_t>>;
    filter_data_t filter_data(filter.word_cnt() + 1024, 0, allocator); // + x to avoid buffer overrun bug in CF
    
    // Build Filter by inserting keys.
    std::size_t n = 0;
    for (std::size_t i = 0; i < to_insert.size(); ++i) {
        const auto key = to_insert[i];
        filter.insert(&filter_data[0], key);
        if (!filter.contains(&filter_data[0], key)) {
            std::cerr << "Breaking..." << std::endl;
            break;
        } else {
            n++;
        }
    }
    // validation (scalar code)
    if (n == 0) {
         std::cerr << "Breaking..." << std::endl;
        std::cerr << "Empty filter?!" << std::endl;
        std::exit(1);
    }
    // // cpu BF
    // {
    //     std::size_t match_cnt = 0;
    //     auto start_probe = std::chrono::high_resolution_clock::now();
    
    //     std::vector<$u32> match_pos(to_lookup.size(), 0);
    //     uint32_t* matches = &match_pos[0];
    //     uint32_t* probe = &to_lookup[0];
    //     filter.batch_contains(&filter_data[0], &to_lookup[0], to_lookup.size(), &match_pos[0], 0);

    //     auto end_probe = std::chrono::high_resolution_clock::now();
    //     std::chrono::duration<double> duration_probe = end_probe - start_probe;
    //     auto probe_time = static_cast<u64>(duration_probe.count());
    //     auto probe_throughput = static_cast<u64>(n / duration_probe.count());

    //     results << "CPU" << "|" << m  << "|" << probe_time << "|" << probe_throughput << '\n';

    //     // CUDA filter
    // }
    //GPU Filters
    cuda_filter<filter_t> cf(filter, &filter_data[0], filter_data.size());
    // naive
    {
        std::vector<$u32> result_bitmap;
        result_bitmap.resize((n), 0);
        typename cuda_filter<filter_t>::perf_data_t perf_data;
        // probe filter
        cf.contains_naive(&to_lookup[0], to_lookup.size(), &result_bitmap[0], perf_data);
        results << "GPU-Naive" << "|" << m  << "|" << perf_data.probe_time << "|" << perf_data.total_throughput << '\n';
    }
    // clustering
    {
        std::vector<$u32> result_bitmap;
        result_bitmap.resize((n), 0);   
        // probe filter
        typename cuda_filter<filter_t>::perf_data_t perf_data;
        cf.contains_clustering(&to_lookup[0], to_lookup.size(), &result_bitmap[0], perf_data, bits_to_sort);
        results << "GPU-Clustering" << "|" << m  << "|" << perf_data.probe_time << "|" << perf_data.total_throughput << std::endl;
    }

}
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//
int main(int argc, char** argv) {
    std::string file_name("results");
    size_t bloom_size = 64ull * 1024ull * 1024ull * 8ull; // 64Mbits
    size_t bits_to_sort = 6; // 64Mbits
    for (int i = 1; i < argc; i++) {
        auto arg = std::string(argv[i]);
        if (arg.substr(0, 2) != "--") {
            exit(EXIT_FAILURE);
        }
        arg = arg.substr(2);
        auto p = split_once(arg, '=');
        auto &arg_name = p.first;
        auto &arg_value = p.second;
        if (arg_name.compare("result_file") == 0) {
            file_name = arg_value;
        }
        if(arg_name.compare("bf_size") == 0) {
            bloom_size = std::stoll(arg_value);
        }
        if(arg_name.compare("bits_to_sort") == 0) {
            bits_to_sort = std::stoll(arg_value);
        }
    }
    std::ofstream result_file;
    result_file.open(std::string(file_name + ".csv"));
    result_file << "Probe Type | Bloom Filter Size | Probe time(s) | Total throughput" << '\n';
    std::size_t insert_cnt = 1<<24; //10M
    std::size_t lookup_cnt = 1<<27; //100M
    std::vector<uint32_t> to_insert(insert_cnt);
    std::vector<uint32_t> to_lookup(lookup_cnt);

    //===----------------------------------------------------------------------===//
    //Benchmark set up
    using key_t = $u32;
    set_uniform_distributed_values(to_insert);
    set_uniform_distributed_values(to_lookup);
    // Data generation.
    benchmark<32, 1, 2>(bloom_size, to_insert, to_lookup, bits_to_sort, result_file);
    result_file.close();
    return 0;
}
