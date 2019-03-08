#pragma once
#include <thread>
#include <vector>
#include <atomic>
#include "bloomfilter/bloom_cuda_filter.cuh"

constexpr gpu_morsel_size  = 100 * 1024 * 1024; 
constexpr cpu_morsel_size  = 10 * 1024; 

struct InflightProbe {
    using probe_t = cuda_filter::probe;
    probe_t probe;
    int64_t num;
    int64_t offset;
    cudaStream_t stream;

    InflightProbe() {
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        probe = probe(num, offset, stream);
    }

    bool is_done() {
        return probe.is_done();
    }

    ~InflightProbe() {
        cudaStreamDestroy(stream);
    }
};

struct Pipeline {
    std::vector<HashTablinho*> hts;
    Table& table; //!< Probe relation
};

struct WorkerThread {
    std::thread worker;
    int device{-1};
    uint32_t hashs[kVecSize];
    bool matches[kVecSize];
    int sel1[kVecSize];
    int sel2[kVecSize];
    int64_t ksum = 0;

    int32_t payload[kVecSize*16];

    Pipeline& pipeline;

    HashTablinho::StaticProbeContext<kVecSize> ctx;

    WorkerThread(bool is_gpu_active, Pipeline& pipeline)
        : pipeline(pipeline) {
        if(is_gpu_active) {
            device = 0;
            worker(execute_pipeline);
        }
    }
    NO_INLINE void execute_pipeline() {
        int64_t morsel_size; 
        auto &table = pipeline.table;
        std::vector<InflightProbe> inflight_probes;

        if (device >= 0) {
            for (int i=0; i<4; i++) {
                probes.emplace_back(InflightProbe());
            }
        }

        while(1) {
            int64_t num = 0;
            int64_t offset = 0;
            size_t finished_probes=0;

            for(auto &probe : inflight_probes) {
                if(probe.is_done()) {
                    auto results = probe.probe.result();
                    do_cpu_join(results, probe.num, probe.offset);
                    finished_probes++;
                    morsel_size = gpu_morsel_size;
                    auto success = table.get_range(num, offset, morsel_size);
                    if (!success)
                        break;
                    do_gpu_work(probe.probe, offset, num);
                }
            }
            if(finished_probes == 0) {
                morsel_size = cpu_morsel_size;
                auto success = table.get_range(num, offset, morsel_size);
                if (!success)
                    break;
                do_cpu_work(table, num, offset);
            }
        }
        for(auto &probe : inflight_probes){
            probe.wait();
            auto results = probe.probe.result();
            do_cpu_join(table, results, probe.num, probe.offset);
        }
    }

    NO_INLINE void do_cpu_work(Table& table, int64_t num, int64_t offset) {
        int* sel = nullptr;

        // TODO: CPU bloom filter

        do_cpu_join(table, nullptr, sel, num, offset);
    }

    NO_INLINE void do_cpu_join(Table& table, int32_t* bf_results, int* sel,
            int64_t num, int64_t offset) {
        assert(!sel == !!bf_results);

        if (sel) {
            assert(num <= kVecSize);
        }

        Vectorized::chunk(offset, num, [&] (auto offset, auto num) {
            int32_t* tkeys = (int32_t*)table.columns[0];
            auto keys = &tkeys[offset];

            if (bf_results) {
                int n = num + (8-1);
                n /= 8;
                n *= 8;
                num = Vectorized::select_match_bit(sel1, bf_results + offset/8, n);
                sel = &sel2[0];
            } else {
                sel = nullptr;
            }

            // probe
            Vectorized::map_hash(hashs, keys, sel, num);

            for (auto ht : pipeline.hts) {
                ht->Probe(ctx, matches, keys, hashs, sel, num);
                num = Vectorized::select_match(sel1, matches, sel, num);
                sel = &sel1[0];

                // TODO: gather some payload columns
                for (int i=1; i<4; i++) {
                    Vectorized::gather_ptr<int32_t>(payload + (i-1)*kVecSize,
                        ctx.tmp_buckets, i, sel, num);
                }
            }

            // global sum
            Vectorized::glob_sum(&ksum, keys, sel, num);
        });
    }

    NO_INLINE void do_gpu_work(probe, offset, num) {
        probe.contains(offset, num);
    }
};

class TaskManager {
public:
    void execute_query(Pipeline& pipeline) {
        std::vector<WorkerThread> workers;

        for(int i = 0; i != std::thread::hardware_concurrency; ++i) {
            workers.emplace_back(WorkerThread(true, pipeline));
        }
        for(auto &worker : workers) {
            worker.thread.join();
        }
       
    }
};