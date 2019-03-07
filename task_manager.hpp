#pragma once
#include <thread>
#include <vector>
#include <atomic>
#include "bloom_cuda_filter.cuh"

constexpr gpu_morsel_size  = 100 * 1024 * 1024; 
constexpr cpu_morsel_size  = 10 * 1024; 

struct InflightProbe {
    using probe_t = cuda_filter::probe;
    probe_t probe;
    int64_t num;
    int64_t offset;

    InflightProbe() {
        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        probe = probe(num, offset, stream);
    }

    bool is_done() {
        return probe.is_done();
    }
}

struct WorkerThread {
    std::thread worker;
    int device{-1};

    WorkerThread(bool is_gpu_active) {
        if(is_gpu_active) {
            device = 0;
            worker(execute_pipeline);
        }
    }
    void execute_pipeline(Pipeline pipeline) {
        int64_t morsel_size; 
        auto &table = pipeline.table;
        std::vector<InflightProbe> probes;

        probes.emplace_back(InflightProbe());
        while(1) {
            int64_t num = 0;
            int64_t offset = 0;

            for(auto &probe : probes) {
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
                do_cpu_work(offset);
            }
        }
        for(auto &probe : probes){
            probe.wait();
            auto results = probe.probe.result();
            do_cpu_join(results, probe.num, probe.offset);
        }

    }

    void do_cpu_join(int32_t results, int64_t num, int64_t offset) {


    }
    void do_gpu_work(probe, offset, num) {
        probe.contains(offset, num);
    }
};

class TaskManager {
public:
    TaskManager() {
       
    
    }
    void execute_query() {

       for(int i = 0; i != std::thread::hardware_concurrency; ++i) {
           workers.emplace_back(WorkerThread(true));
       }
       for(auto &worker : workers) {
           worker.thread.join();
       }
       
    }

private:
    std::vector<WorkerThread> workers;
    std::atomic<size_t> global_counter;
};