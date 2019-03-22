#pragma once

#include <fstream>

struct ProfilePrinter {
	ProfilePrinter(size_t repetitions) : repetitions(repetitions){}

	void write_header (std::ofstream& os) {
		os << "Pipeline Time"  	  << "|";
		os << "CPU Join Time"  	  << "|";
		os << "GPU Probe Time" 	  << "|";
		os << "CPU/GPU Time" 	  << "|";
		os << "Pre Filter Tuples" << "|";
		os << "Filtered Tuples"   << "|";
		os << "Pos Join Tuples"   << '\n';

	}

	void write_profile(std::ofstream& os) {
		os << pipeline_time 	<< "|";
		os << cpu_time 			<< "|";
		os << gpu_time 			<< "|";
		os << cpu_gpu_time 		<< "|";
		os << pre_filter_tuples << "|";
		os << fitered_tuples 	<< "|";
		os << pos_join_tuples 	<< "\n";

	}

	int64_t pipeline_time{0};
	int64_t cpu_time{0};
	int64_t gpu_time{0};
	int64_t cpu_gpu_time{0};
	int64_t pre_filter_tuples{0};
	int64_t fitered_tuples{0};
	int64_t pos_join_tuples{0};
	size_t repetitions{0};

};