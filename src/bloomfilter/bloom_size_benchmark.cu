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
        const auto hash_value = filter.hash(key);
        filter.insert(&filter_data[0], key);
        if (!filter.contains(&filter_data[0], key)) {
            std::cerr << "Breaking..." << std::endl;
            break;
        } else if (!filter.contains_with_hash(&filter_data[0], hash_value, key)) {
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
    std::size_t matches = 0, matches_naive = 0;
    for(std::size_t i = 0; i != to_lookup.size(); ++i) {
        const auto key = to_lookup[i];
        const auto hash_val = filter.hash(key);
        if (filter.contains(&filter_data[0], key)) {

            matches_naive++;
        }
        if(filter.contains_with_hash(&filter_data[0], hash_val, key)) {
            matches++;
        }
    }
    assert(matches == matches_naive);
    // // cpu BF
    //GPU Filters
    cuda_filter<filter_t> cf(filter, &filter_data[0], filter_data.size());
    // naive
/*    {
        std::vector<$u32> result_bitmap;
        result_bitmap.resize((n), 0);
        typename cuda_filter<filter_t>::perf_data_t perf_data;
        // probe filter
        cf.contains_naive(&to_lookup[0], to_lookup.size(), &result_bitmap[0], perf_data);
        results << "GPU-Naive" << "|" << m  << "|" << perf_data.probe_time << "|" << perf_data.total_throughput << '\n';
    	std::cout << "GPU-Naive" << "|" << m  << "|" << perf_data.probe_time << "|" << perf_data.total_throughput << '\n';
        for(size_t i = 0; i != n; ++i ) {
            if(result_bitmap[i] != 0) {
                std::cout << result_bitmap[i] << std::endl;
            }
        }
	}*/
        std::cout << "Clustering" << std::endl;
    // clustering
    {
        std::vector<$u32> result_bitmap;
        result_bitmap.resize((to_lookup.size()), 0);
        // probe filter
        typename cuda_filter<filter_t>::perf_data_t perf_data;
        cf.contains_clustering(&to_lookup[0], to_lookup.size(), &result_bitmap[0], perf_data, bits_to_sort);
        std::cout << "GPU-Clustering" << "|" << m  << "|" << perf_data.probe_time << "|" << perf_data.total_throughput << '\n';
        results << "GPU-Clustering" << "|" << m  << "|" << perf_data.probe_time << "|" << perf_data.total_throughput << std::endl;
        
        size_t count = 0;
        for(size_t i = 0; i != to_lookup.size(); ++i ) {
            if(result_bitmap[i] != 0) {
                //std::cout << result_bitmap[i] << std::endl;
                count++;
            }
        }
        std::cout << "possible matches found " << count << " - matches found " << matches  << " total " << to_lookup.size() - matches << std::endl;
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
    // Parse args
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
    std::size_t lookup_cnt = 1<<28; //100M
    std::vector<uint32_t> to_insert(insert_cnt);
    std::vector<uint32_t> to_lookup(lookup_cnt);

    //===----------------------------------------------------------------------===//
    //Benchmark set up
    using key_t = $u32;
    set_uniform_distributed_values(to_insert);
    set_uniform_distributed_values(to_lookup);

    //std::generate(to_lookup.begin(), to_lookup.end(), [n = 0]() mutable {return ++n;});


    // Data generation.
    benchmark<32, 1, 2>(bloom_size, to_insert, to_lookup, bits_to_sort, result_file);
    result_file.close();
    return 0;
}
