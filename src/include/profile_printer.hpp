#pragma once

#include <fstream>

struct ProfilePrinter {
	ProfilePrinter(size_t repetitions) : repetitions(repetitions){}

	void write_header (std::ofstream& os) {
		os << "Pipeline Time"  	  << "|";
		os << "CPU Join Time"  	  << "|";
		os << "GPU Probe Time" 	  << "|";
		os << "CPU/GPU Time" 	  << "|";
#ifdef PROFILE
		os << "Pre Filter Tuples" << "|";
		os << "Filtered Tuples"   << "|";
		os << "Pre Join Tuples"   << "|";
		os << "Pos Join Tuples"   << "|";
#endif
		os << "Selectivity"       << '\n';

	}

	void write_profile(std::ofstream& os) {
		os << (pipeline_time 	 / repetitions)	<< "|";
		os << (cpu_time 	 	 / repetitions)	<< "|";
		os << (gpu_time 	 	 / repetitions)	<< "|";
		os << (cpu_gpu_time 	 / repetitions)	<< "|";
#ifdef PROFILE
		os << (pre_filter_tuples / repetitions) << "|";
		os << (fitered_tuples    / repetitions)	<< "|";
		os << (pre_join_tuples   / repetitions)	<< "|";
#endif
		os << (pos_join_tuples   / repetitions)	<< "|";
		os << (selectivity)						<< "\n";

	}

	int64_t pipeline_time{0};
	int64_t cpu_time{0};
	int64_t gpu_time{0};
	int64_t cpu_gpu_time{0};
	int64_t pre_filter_tuples{0};
	int64_t fitered_tuples{0};
	int64_t pos_join_tuples{0};
	size_t selectivity{0};
	size_t pre_join_tuples{0};
	size_t repetitions{0};

};