#include "hash_table.hpp"
#include "bloomfilter.hpp"
#include "task_manager.hpp"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>
#include "bloomfilter/util.hpp"


//===----------------------------------------------------------------------===//
int main(int argc, char** argv) {
    auto params = parse_command_line(argc, argv);
    TaskManager manager;

    auto build_size = ((params.selectivity * params.probe_size) / 100);
    std::cout << " Probe Size: " << params.probe_size << " -- Build Size: " << build_size << " -- Selectivity: " << params.selectivity << std::endl;

    Table table_build(1,build_size);
    populate_table(table_build);

    auto ht = new HashTablinho(
        sizeof(int32_t) + // key 
        NUM_PAYLOAD * sizeof(int32_t), // payload cols
        build_size);

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

    Table table_probe(1,params.probe_size);
    populate_table(table_probe);
    std::vector<HashTablinho*> hts = {ht};
    Pipeline pipeline(hts, table_probe, params);
    //manager.execute_query(pipeline);

    // Build 128 bytes Blocked Bloom Filter on CPU
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
        }
        auto final_elapsed_time = total_seconds / params.num_repetitions;
        std::cout << " Probe time (sec):" << final_elapsed_time << std::endl;
    }

    return 0;
}
//===----------------------------------------------------------------------===//