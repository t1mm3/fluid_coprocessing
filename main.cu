#include "hash_table.hpp"
#include "profile_printer.hpp"
#include "bloomfilter.hpp"
#include "task_manager.hpp"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>
#include <iostream>
#include "bloomfilter/util.hpp"


void gen_csv(const std::string& fname, const Table& t, bool probe) {
    uint32_t *table_keys = (uint32_t *)t.columns[0];
    std::ofstream f;
    f.open(fname);

    auto num = t.size();
    for (size_t row=0; row<num; row++) {
        f << table_keys[row];
        if (!probe) {
            for (int k=0; k<NUM_PAYLOAD; k++) {
                f << "|";
                f << "0";
            }
        }
        f << "\n";
    }
    f.close();
};

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
size_t file_size(const std::string& fname) {
    struct stat st;
    if (!stat(fname.c_str(), &st))  {
        return st.st_size;
    }
    return 0;
}

void write_column(const std::string& file, Table& table, size_t col, size_t num) {
    std::ofstream out(file, std::ios::out | std::ios::binary);
    assert(out.is_open());
    int32_t *d = (int32_t *)table.columns[col];

    out.write((char*)d, sizeof(int32_t) * num);
    out.close();
};

void write_ksum(const std::string& file, int64_t ksum) {
    std::ofstream out(file, std::ios::out | std::ios::binary);
    assert(out.is_open());
    out.write(reinterpret_cast<const char *>(&ksum), sizeof(ksum));
    out.close();
};

void read_sum(const std::string& file, int64_t& ksum) {
    std::ifstream in(file, std::ios::in | std::ios::binary);
    assert(in.is_open());

    in.read(reinterpret_cast<char *>(&ksum), sizeof(int64_t));
    in.close();
};

void read_column(Table& table, const std::string& file, size_t col, size_t num) {
    std::ifstream in(file, std::ios::in | std::ios::binary);
    assert(in.is_open());

    int32_t *d = (int32_t *)table.columns[col];

    in.read((char*)d, sizeof(int32_t) * num);

    in.close();
};

#include <tuple>
#include <sstream>

void test() {
    int sel[kVecSize];
    uint8_t bit;
    int num;
    uint32_t bit32;

    bit = 0xFF;
    num = Vectorized::select_match_bit(true, sel, &bit, 8);
    assert(num == 8);
    for (int i=0; i<num; i++) {
        assert(sel[i] == i);
    }

    bit = 1;
    num = Vectorized::select_match_bit(true, sel, &bit, 8);
    assert(num == 1);
    assert(sel[0] == 0);

    bit32 = 0;
    int exp_num = 0;
    for (int i=0; i<32; i++) {
        if (i % 4 == 0) {
            bit32 |= 1 << i;
            exp_num++;
        } 
    }

    memset(sel, 0, sizeof(sel));
    num = Vectorized::select_match_bit(true, sel, (uint8_t*)&bit32, 30);
    assert(8 == num);
    assert(num == exp_num);
    assert(sel[0] == 0);
    assert(sel[1] == 4);
    assert(sel[2] == 8);
    assert(sel[3] == 12);
    assert(sel[4] == 16);
    assert(sel[5] == 20);
    assert(sel[6] == 24);
    assert(sel[7] == 28);
}
//===----------------------------------------------------------------------===//
int main(int argc, char** argv) {
    test();
    auto params = parse_command_line(argc, argv);
    TaskManager manager;
    std::ofstream results_file;
    results_file.open("results.csv", std::ios::out);

    std::cout << " Probe Size: " << params.probe_size << " -- Build Size: " << params.build_size << std::endl;

    size_t build_size = params.build_size;
    size_t probe_size = params.probe_size;
    size_t selectivity = params.selectivity;
    size_t num_columns = params.num_columns;
    Table table_build(num_columns, build_size);
    Table table_probe(1,probe_size);

    auto gen_fname = [&] (size_t id) {
        std::ostringstream s;
        s << "data_" << id << "_" << "_s_" << selectivity
            << "_b_" << build_size << "_p_" << probe_size << ".bin";
        return s.str();
    };

    const std::string bfile(gen_fname(0));
    const std::string pfile(gen_fname(1));
    const std::string ksum(gen_fname(3));
    bool cached = true;

    if (!file_size(bfile) || !file_size(pfile) || !file_size(ksum)) {
        std::cout << "Files not cached. Recreating ..." << std::endl;
        // not cached, create files
        cached = false;

        populate_table(table_build);
        populate_table(table_probe);

        set_selectivity(table_build, table_probe, selectivity);
        auto expected_ksum = calculate_matches_sum(table_build, table_probe, selectivity);
        std::cout << "Writing ksum to disk ..." << std::endl;
        write_ksum(ksum, expected_ksum);

        std::cout << "Writing 'build' to disk ..." << std::endl;
        write_column(bfile, table_build, 0, build_size);
        std::cout << "Writing 'probe' to disk ..." << std::endl;
        write_column(pfile, table_probe, 0, probe_size);

        std::cout << "Done" << std::endl;
    }

    if (params.only_generate) {
        exit(0);
    }

    // load data
    assert(file_size(bfile) > 0);
    assert(file_size(pfile) > 0);
    assert(file_size(ksum) > 0);
    assert(file_size(ksum) == sizeof(int64_t));
    assert(file_size(bfile) == sizeof(int32_t) * build_size);
    assert(file_size(pfile) == sizeof(int32_t) * probe_size);

    read_column(table_build, bfile, 0, build_size);
    read_column(table_probe, pfile, 0, probe_size);
    int64_t expected_ksum = 0;
    read_sum(ksum, expected_ksum);


    assert(params.gpu_morsel_size >= params.cpu_morsel_size);


    if (!params.csv_path.empty()) {
        std::cout << "Writing build relation ..." <<std::endl;

        gen_csv(params.csv_path + "build.csv", table_build, false);

        std::cout << "Writing probe relation ..." <<std::endl;
        gen_csv(params.csv_path + "probe.csv", table_probe, true);

        std::cout << "Done" << std::endl;
        exit(0);
    }

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

    std::vector<HashTablinho*> hts = {ht};
    Pipeline pipeline(hts, table_probe, params);

    // Build Blocked Bloom Filter on CPU (Block size = 128 Bytes)
    {
        size_t m = params.filter_size;
        FilterWrapper filter(m);
        uint32_t *table_keys = (uint32_t *)table_build.columns[0];
        uint32_t *probe_keys = static_cast<uint32_t*>(table_probe.columns[0]);
        std::set<uint32_t> positions;

        for (std::size_t i = 0; i < table_build.size(); ++i) {
            const auto key = (uint32_t)*(table_keys + i);
            //std::cout << "Insert key " << key << " position " << i << '\n';
            filter.insert(key);
        }

        // Validate Filter on CPU
        for (std::size_t i = 0; i < table_build.size(); ++i) {
            const auto key = (uint32_t)*(table_keys + i);
            auto match = filter.contains(key);
            if(!match)
                std::cout << "no match key " << key << " position "<< i << '\n';

        }
        std::cout << std::endl;

        // cuda instance of bloom filter logic on GPU
        FilterWrapper::cuda_filter_t cf(filter.bloom_filter, &(filter.filter_data[0]), filter.bloom_filter.word_cnt());


        ProfilePrinter profile_info(params);
        profile_info.write_header(results_file);

        for(auto i = 0; i < params.num_repetitions + params.num_warmup; ++i) {
            //execute probe
            const auto start = std::chrono::system_clock::now();
            const auto start_cycles = rdtsc();
            manager.execute_query(pipeline, filter, cf, profile_info);
            auto end_cycles = rdtsc();
            auto end = std::chrono::system_clock::now();

            if (i >= params.num_warmup) {
                // Profile output
                profile_info.pipeline_cycles += (double)(end_cycles - start_cycles);
                profile_info.pipeline_sum_thread_cycles += (double)(pipeline.prof_pipeline_cycles.cycles);
                profile_info.pipeline_time   += std::chrono::duration<double>(end - start).count();
                profile_info.cpu_time        += (double)pipeline.prof_aggr_cpu.cycles;
                profile_info.cpu_join_time   += (double)pipeline.prof_join_cpu.cycles;
                profile_info.cpu_expop_time  += (double)pipeline.prof_expop_cpu.cycles;  
                profile_info.gpu_time        += (double)pipeline.prof_aggr_gpu.cycles;
                profile_info.cpu_gpu_time    += (double)pipeline.prof_aggr_gpu_cpu_join.cycles;

#ifdef PROFILE
                profile_info.pre_filter_tuples += pipeline.num_prefilter;
                profile_info.fitered_tuples    += pipeline.num_postfilter;
                profile_info.pos_join_tuples   += pipeline.num_prejoin;
                profile_info.pos_join_tuples   += pipeline.num_postjoin;
#endif
            }
            if(expected_ksum != pipeline.ksum) {
                std::cout << " invalid ksum:" << pipeline.ksum << " expected:" << expected_ksum << std::endl;
            }
            pipeline.reset();
        }
        double final_elapsed_time = profile_info.pipeline_time / (double)params.num_repetitions;
        std::cout << " Probe time (sec):" << final_elapsed_time << std::endl;

        profile_info.write_profile(results_file);
    }
    results_file.close();

    return 0;
}
//===----------------------------------------------------------------------===//