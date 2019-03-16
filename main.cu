#include "hash_table.hpp"
#include "bloomfilter.hpp"
#include "task_manager.hpp"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>






//===----------------------------------------------------------------------===//
void set_uniform_distributed_values(int32_t* column, size_t range_size) {
    //thread_local allows unique seed for each thread
    thread_local std::random_device rd;     // Will be used to obtain a seed for the random number engine
    thread_local std::mt19937 engine(rd()); //Standard mersenne_twister_engine seeded with rd()

    std::uniform_int_distribution<int32_t> distribution;
    auto sampler = [&]() { return distribution(engine); };   //Use distribution to transform the random unsigned int generated by engine into an int in [0, u32]
    auto increment_one = [n = 0]() mutable { return ++n; };   //Use distribution to transform the random unsigned int generated by engine into an int in [0, u32]
    std::generate(&column[0], column + range_size, increment_one); // Initializes the container with random uniform distributed values
}
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
void populate_table(Table &table) {
    for(auto &column : table.columns)
        set_uniform_distributed_values((int32_t*)column, table.size());
}
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
int main() {


    TaskManager manager;

    Table table_build(1,1024);
    populate_table(table_build);

    auto ht = new HashTablinho(4+4*4, TABLE_SIZE);

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

    Table table_probe(1,16*1024);
    populate_table(table_probe);
    Pipeline pipeline = { {ht}, table_probe};
    //manager.execute_query(pipeline);

    // Build 128 bytes Blocked Bloom Filter on CPU
    {
        size_t m = 64*8*1024*1024;
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
        //execute probe
        auto start = std::chrono::system_clock::now();
        manager.execute_query(pipeline, filter, cf);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        std::cout << " Probe time:" << elapsed_seconds.count() << std::endl;
    }

    return 0;
}
//===----------------------------------------------------------------------===//