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
        auto hash_key = filter.hash(key);
        if (!filter.contains_with_hash(&filter_data[0], hash_key, key)) {
            break;
        } else {
            n++;
        }
    }
    // validation (scalar code)
    if (n == 0) {
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

    // CUDA filter
    cuda_filter<filter_t> cf(filter, &filter_data[0], filter_data.size(),nullptr,0);

         // probe the filter
    {
        std::vector<$u32> result_bitmap;
        result_bitmap.resize((n), 0);
    
        // probe filter
        typename cuda_filter<filter_t>::perf_data_t perf_data;
        // TODO (HL) result bitmaps sizing issue with the sorted_kernel!!!

        cf.contains_clustering(&to_lookup[0], to_lookup.size(), &result_bitmap[0], perf_data, bits_to_sort);
    
        std::cout  << "=============================== "                                << '\n'
                  << " Results: "                                                       << '\n'
                  << " Word count: "                 << word_cnt                        << '\n'
                  << " Block size: "                 << word_cnt * 4                    << '\n'
                  << " Zone count: "                 << zone_cnt                        << '\n'
                  << " k: "                          << k                               << '\n'
                  << " Bloom filter size(MiB): "     << m/(8*1024*1024)                 << '\n'
                  << " Lookup size: "                << to_lookup.size()                << '\n'
                  << " Blocks: "                     << perf_data.cuda_block_cnt        << '\n'
                  << " CUDA Block size: "            << perf_data.cuda_block_size       << '\n'
                  << " Hash throughput: "            << perf_data.hash_throughput       << '\n'
                  << " Hash time (ms): "             << perf_data.hash_time * 1000      << '\n'
                  << " Sort throughput: "            << perf_data.sort_throughput       << '\n'
                  << " Sort time (ms): "             << perf_data.sort_time * 1000      << '\n'
                  << " Probes per second: "          << perf_data.probes_per_second     << '\n'
                  << " Probe time (ms): "            << perf_data.probe_time * 1000     << '\n'
                  << " Candidate List time (ms): "   << perf_data.candidate_time * 1000 << '\n'
                  << " Total throughput: "           << perf_data.total_throughput      << '\n'
                  << "=============================== "                                 << '\n'
                  << std::endl;
        
        results << m/(8*1024*1024)             << ';';
        results <<  word_cnt * 4               << ';';
        results << bits_to_sort                << ';';
        results << to_insert.size()            << ';';
        results << perf_data.hash_time * 1000  << ';';
        results << perf_data.sort_time * 1000  << ';';
        results << perf_data.probe_time * 1000 << ';';
        results << perf_data.total_throughput  << "\n";

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
    std::string result_file("bench_bits.csv"), output_file("output.txt");
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
        result_file = arg_value;
      }
      if (arg_name.compare("output_file") == 0) {
        output_file = arg_value;
      }
    }

    std::ofstream out(std::string("results/" + output_file));
    std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf
    std::cout.rdbuf(out.rdbuf()); //redirect std::cout to out.txt!
    std::ofstream results;
    results.open(std::string("results/" + result_file));
    results << "Bloom filter size (MiB); Block size (bytes); bits to sort; Probe size ; Hash time (ms); Sort time (ms); Probe time (ms); Total throughput" << '\n';
    //get_device_properties();

    //===----------------------------------------------------------------------===//
    //Benchmark set up
    auto increment_one = [n = 0]() mutable {return ++n;};
    std::vector<size_t> bits_to_sort(32);
    std::generate(bits_to_sort.begin(), bits_to_sort.end(), increment_one);
    std::size_t default_m = 1ull * 1024ull * 1024ull * 8ull; // 256MiB
    auto m = { default_m, default_m * 2, default_m * 4, default_m * 8, default_m * 16, default_m * 32, default_m * 64, default_m * 128, default_m * 256, default_m * 512};
    auto input_size = {1ull<<28}; // 10K 100K 1M 10M 100M
    
    // Data generation.
    using key_t        = $u32;

    for(auto to_insert_cnt : input_size){
        const std::size_t to_lookup_cnt = to_insert_cnt;
        std::vector<key_t> to_insert(to_insert_cnt);
        std::vector<key_t> to_lookup(to_lookup_cnt);
  
        for(auto bloom_size : m) {
            set_random_values(to_insert);
            set_random_values(to_lookup);
            auto bf_size = (bloom_size / (8*1024*1024));
  
            for(auto& bits : bits_to_sort) {
                std::cout << "to_insert.size(): " << to_insert_cnt/1024    << " K-keys" << std::endl;
                std::cout << "to_lookup.size(): " << to_insert_cnt/1024    << " K-keys" << std::endl;
                std::cout << "bits to sort: "     << bits                  << " bits"   << std::endl;
                std::cerr << "bits to sort: "     << bits                  << " bits"   << std::endl;
                std::cerr << "Bloom Size: "       << bf_size               << " MiB"    << std::endl;
                //Register Blocking
                //benchmark<8, 1, 2>(bloom_size, to_insert, to_lookup, bit);
                //benchmark<16, 1, 2>(bloom_size, to_insert, to_lookup, bit);
                benchmark<32, 1, 2>(bloom_size, to_insert, to_lookup, bits, results);
                //benchmark<64, 1, 2>(bloom_size, to_insert, to_lookup, bit);
            }
        }
    }
    results.close();

    return 0;
}
