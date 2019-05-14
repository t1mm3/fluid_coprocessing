#pragma once

#include <fstream>
#include "bloomfilter/util.hpp"

struct ProfilePrinter {
	const params_t& params;
	const int64_t repetitions;

	ProfilePrinter(const params_t& params) : repetitions(params.num_repetitions), params(params) {}

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

		os << "CPUBloomFilter"	<< "|";
		os << "FilterSize"		<< "|";
		os << "Slowdown"		<< "|";
		os << "CPUMorselSize"	<< "|";
		os << "GPUMorselSize"	<< "|";
		os << "Selectivity"		<< '|';
		os << "Streams"		<< '|';
		os << "TuplesGpuProbe"	<< "|";
		os << "TuplesGpuConsume"	<< "\n";

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
		os << (params.cpu_bloomfilter) 			<< "|";
		os << (params.filter_size)				<< "|";
		os << (params.slowdown)					<< "|";
		os << (params.cpu_morsel_size)			<< "|";
		os << (params.gpu_morsel_size)			<< "|";
		os << (params.selectivity)				<< "|";
		os << (params.num_gpu_streams)				<< "|";
		os << (tuples_gpu_probe  / repetitions)			<< "|";
		os << (tuples_gpu_consume  / repetitions)				<< "\n";
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
	int64_t tuples_gpu_probe{0};
	int64_t tuples_gpu_consume{0};

	size_t pre_join_tuples{0};
	uint64_t semijoin_time {0};

};