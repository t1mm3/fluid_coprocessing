#pragma once

#include <fstream>

struct ProfilePrinter {
	ProfilePrinter(size_t repetitions) : repetitions(repetitions){}

	void write_header (std::ofstream& os) {
		os << "Pipeline Cycles"   << "|";
		os << "Pipeline Sum Thread Cycles" << "|";
		os << "Pipeline Time"  	  << "|";
		os << "CPU Time"  	  << "|";
		os << "CPU Join Time" << "|";
		os << "CPU ExpOp Time" << "|";
		os << "GPU Probe Time" 	  << "|";
		os << "CPU/GPU Time" 	  << "|";
#ifdef PROFILE
		os << "Pre Filter Tuples" << "|";
		os << "Filtered Tuples"   << "|";
		os << "Pre Join Tuples"   << "|";
		os << "Pos Join Tuples"   << "|";
#endif

		os << "CPUBloomFilter" << "|";
		os << "Selectivity"       << '\n';

	}

	void write_profile(std::ofstream& os) {
		os << (pipeline_cycles / repetitions) << "|";
		os << (pipeline_sum_thread_cycles / repetitions) << "|";
		os << (pipeline_time 	 / repetitions)	<< "|";
		os << (cpu_time 	 	 / repetitions)	<< "|";
		os << (cpu_join_time	 / repetitions)	<< "|";
		os << (cpu_expop_time	 / repetitions)	<< "|";
		os << (gpu_time 	 	 / repetitions)	<< "|";
		os << (cpu_gpu_time 	 / repetitions)	<< "|";
#ifdef PROFILE
		os << (pre_filter_tuples / repetitions) << "|";
		os << (fitered_tuples    / repetitions)	<< "|";
		os << (pre_join_tuples   / repetitions)	<< "|";
#endif
		os << (pos_join_tuples   / repetitions)	<< "|";
		os << (cpu_bloomfilter) 				<< "|";
		os << (selectivity)						<< "\n";

	}

	double pipeline_cycles{0};
	double pipeline_time{0};
	double pipeline_sum_thread_cycles{0};
	double cpu_time{0};
	double cpu_join_time{0};
	double cpu_expop_time{0};
	double gpu_time{0};
	double cpu_gpu_time{0};
	int64_t pre_filter_tuples{0};
	int64_t fitered_tuples{0};
	int64_t pos_join_tuples{0};

	int cpu_bloomfilter{0};
	size_t selectivity{0};
	size_t pre_join_tuples{0};
	double repetitions{0};

};