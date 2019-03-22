#include "hash_table.hpp"
#include "bloomfilter.hpp"
#include "task_manager.hpp"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>
#include <iostream>
#include "bloomfilter/util.hpp"


//===----------------------------------------------------------------------===//
int main(int argc, char** argv) {
    auto params = parse_command_line(argc, argv);
    TaskManager manager;
    std::ofstream results_file;
    results_file.open("results.csv", std::ios::out);

    std::cout << " Probe Size: " << params.probe_size << " -- Build Size: " << params.build_size << std::endl;

    // Build relation
    Table table_build(params.num_columns, params.build_size);
    populate_table(table_build);

    auto ht = new HashTablinho(
        sizeof(int32_t) + // key 
        NUM_PAYLOAD * sizeof(int32_t), // payload cols
        params.build_size);

     //build table
    uint32_t hashs[kVecSize];
    table_build.chunk([&] (auto columns, auto num_columns, auto offset, auto num) {
        int32_t *table_keys = (int32_t *)table_build.columns[0];
        Vectorized::chunk(offset, num, [&] (auto offset, auto num) {
            auto keys = table_keys+offset;
            
            int* sel = nullptr;
            Vectorized::map_hash(hashs, keys, sel, num);
            ht->Insert(keys, hashs, sel, num);
        }, kVecSize);

        // FIXME: build bloom filter
    }, [&] () {
        // finished
        ht->FinalizeBuild();
    });

    // Probe relation
    Table table_probe(params.num_columns, params.probe_size);
    populate_table(table_probe);

    set_selectivity(table_build, table_probe, params.selectivity);
    std::vector<HashTablinho*> hts = {ht};
    Pipeline pipeline(hts, table_probe, params);

    // Build Blocked Bloom Filter on CPU (Block size = 128 Bytes)
    {
        size_t m = params.filter_size;
        FilterWrapper filter(m);
        uint32_t *table_keys = (uint32_t *)table_build.columns[0];

        for (std::size_t i = 0; i < table_build.size(); ++i) {
            const auto key = (uint32_t)*(table_keys + i);
            //std::cout << "Insert key " << key << " position " << i << '\n';
            filter.insert(key);
        }

        FilterWrapper::cuda_filter_t cf(filter.bloom_filter, &(filter.filter_data[0]), filter.bloom_filter.word_cnt());
        
        for (std::size_t i = 0; i < table_build.size(); ++i) {
            const auto key = (uint32_t)*(table_keys + i);
            auto match = filter.contains(key);
            if(!match)
                std::cout << "no match key " << key << " position "<< i << '\n';

        }
        std::cout << std::endl;

        double total_seconds = 0.0;
        for(auto i = 0; i != params.num_repetitions; ++i) {
            //execute probe
            auto start = std::chrono::system_clock::now();
            manager.execute_query(pipeline, filter, cf);
            auto end = std::chrono::system_clock::now();
            total_seconds += std::chrono::duration<double>(end - start).count();

            pipeline.reset();
        }
        auto final_elapsed_time = total_seconds / params.num_repetitions;
        std::cout << " Probe time (sec):" << final_elapsed_time << std::endl;
        results_file << "Total time :" << final_elapsed_time << std::endl;
    }
    results_file.close();

    return 0;
}
//===----------------------------------------------------------------------===//