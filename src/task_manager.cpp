#include "task_manager.hpp"

void TaskManager::execute_query(Pipeline& pipeline) {
    std::vector<WorkerThread<filter_t, word_t>> workers;

    auto hardware_threads = std::thread::hardware_concurrency();
    assert(hardware_threads > 0);
    
    for(int i = 0; i != hardware_threads; ++i) {
        workers.emplace_back(WorkerThread(false, pipeline));
    }
    for(auto &worker : workers) {
        worker.thread.join();
    }
        
}

void WorkerThread::execute_pipeline() {
        int64_t morsel_size; 
        auto &table = pipeline.table;

#ifdef HAVE_CUDA
    std::vector<InflightProbe> inflight_probes;

    if (device > 0) {
        //instantiate CUDA Filter
        cuda_filter<filter_t> cf(filter, &filter_data[0], filter_data.size());
        for (int i = 0; i < number_of_streams; i++) {
            //create probes
            probes.emplace_back(InflightProbe(cf));
        }
    }
#endif

    while(1) {
        int64_t num = 0;
        int64_t offset = 0;
        size_t finished_probes = 0;

#ifdef HAVE_CUDA
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
#endif
    	if (finished_probes == 0) {
    	    morsel_size = cpu_morsel_size;
    	    auto success = table.get_range(num, offset, morsel_size);
    	    if (!success)
    	        break;
    	    do_cpu_work(table, num, offset);
    	}
	}
#ifdef HAVE_CUDA
    for(auto &probe : inflight_probes){
        probe.wait();
        auto results = probe.probe.result();
        do_cpu_join(table, results, probe.num, probe.offset);
    }
#endif
}

void WorkerThread::do_cpu_join(Table& table, int32_t* bf_results, int* sel,
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
            for (int i = 1; i < 4; i++) {
                Vectorized::gather_ptr<int32_t>(payload + (i-1)*kVecSize,
                    ctx.tmp_buckets, i, sel, num);
            }
        }

        // global sum
        Vectorized::glob_sum(&ksum, keys, sel, num);
    });
}