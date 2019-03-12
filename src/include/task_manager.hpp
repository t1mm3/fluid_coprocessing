#pragma once
#include <thread>
#include <vector>
#include <atomic>
#include "bloomfilter/bloom_cuda_filter.hpp"

constexpr size_t GPU_MORSEL_SIZE    = 100 * 1024 * 1024; 
constexpr size_t CPU_MORSEL_SIZE    = 10 * 1024; 
constexpr size_t NUMBER_OF_STREAMS  = 4; 

#ifdef HAVE_CUDA
struct InflightProbe {
    using cuda_probe_t = typename cuda_filter<filter_t>::probe;
    cuda_probe_t probe;
    int64_t num;
    int64_t offset;
    cudaStream_t stream;

    InflightProbe(const cuda_filter& cf) {
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        probe = probe(cf, batch_size_gpu, stream);
    }

    bool is_done() {
        return probe.is_done();
    }

    ~InflightProbe() {
        cudaStreamDestroy(stream);
    }
};
#endif

struct Pipeline {
    std::vector<HashTablinho*> hts;
    Table& table; //!< Probe relation
};

template <typename filter_t, typename word_t>
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
    const filter_t& filter;
    const word_t*  filter_data;

    HashTablinho::StaticProbeContext<kVecSize> ctx;

    WorkerThread(int gpu_device, Pipeline& pipeline)
            : pipeline(pipeline), device(gpu_device), worker(execute_pipeline, this) {
    }

    WorkerThread(int gpu_device, Pipeline& pipeline, const filter_t& filter, const word_t* __restrict filter_data)
            : pipeline(pipeline), device(gpu_device), worker(execute_pipeline, this), filter(filter) {
    }
    
    NO_INLINE void execute_pipeline();

    NO_INLINE void do_cpu_work(Table& table, int64_t num, int64_t offset) {
        int* sel = nullptr;

        // TODO: CPU bloom filter

        do_cpu_join(table, nullptr, sel, num, offset);
    }

    NO_INLINE void do_cpu_join(Table& table, int32_t* bf_results, int* sel,
            int64_t num, int64_t offset);

#ifdef HAVE_CUDA
    NO_INLINE void do_gpu_work(probe, offset, num) {
        probe.contains(offset, num);
    }
#endif
};

class TaskManager {
public:

    void execute_query(Pipeline& pipeline);

    template <typename filter_t, typename word_t>
    void execute_query(Pipeline& pipeline, const filter_t& filter, const word_t* __restrict filter_data) {
        std::vector<WorkerThread<filter_t, word_t>> workers;

        for(int i = 0; i != std::thread::hardware_concurrency; ++i) {
            workers.emplace_back(WorkerThread<filter_t, word_t>(true, pipeline. filter, filter_data));
        }
        for(auto &worker : workers) {
            worker.thread.join();
        }
       
    }
};