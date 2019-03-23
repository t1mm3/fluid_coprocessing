#pragma once

#include "constants.hpp"

#include <algorithm>
#include <bitset>
#include <cmath>
#include <dtl/batchwise.hpp>
#include <dtl/dtl.hpp>
#include <functional>
#include <random>
#include <string>
#include <unordered_set>
#include <set>

//===----------------------------------------------------------------------===//
using vector_t = std::vector<uint32_t>;

// The (static) unrolling factor
constexpr size_t UNROLL_FACTOR = 16;
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
template <class Container> void set_uniform_distributed_values(Container &container) {
	// thread_local allows unique seed for each thread
	thread_local std::random_device rd;     // Will be used to obtain a seed for the random number engine
	thread_local std::mt19937 engine(rd()); // Standard mersenne_twister_engine seeded with rd()

	std::uniform_int_distribution<typename Container::value_type> distribution;
	auto sampler = [&]() { return distribution(engine); }; // Use distribution to transform the random unsigned int
	                                                       // generated by engine into an int in [0, u32]
	std::generate(std::begin(container), std::end(container),
	              sampler); // Initializes the container with random uniform
	                        // distributed values
}
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
namespace internal {

static void gen_random_data(vector_t &data, const std::size_t element_cnt, const bool unique_elements,
                            const std::function<uint32_t()> &rnd) {
	data.clear();
	data.reserve(element_cnt);

	if (unique_elements) { // Generate unique elements.

		if (element_cnt > (1ull << 32)) {
			std::cerr << "Cannot create more than 2^32 unique integers." << std::endl;
			std::exit(1);
		}

		if (element_cnt == (1ull << 32)) {

			// Entire integer domain.
			for (std::size_t i = 0; i < element_cnt; i++) {
				data.push_back(static_cast<uint32_t>(i));
			}
			std::random_device rnd_device;
			std::shuffle(data.begin(), data.end(), rnd_device);

		} else {

			auto is_in_set = new std::bitset<1ull << 32>;
			std::size_t c = 0;
			while (c < element_cnt) {
				auto val = rnd();
				if (!(*is_in_set)[val]) {
					data.push_back(val);
					(*is_in_set)[val] = true;
					c++;
				}
			}
			delete is_in_set;
		}
	} else { // Generate non-unique elements.
		for (std::size_t i = 0; i < element_cnt; i++) {
			data.push_back(rnd());
		}
	}
}
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
static void gen_random_data_64(std::vector<uint64_t> &data, const std::size_t element_cnt, const bool unique_elements,
                               const std::function<uint64_t()> &rnd) {
	data.clear();
	data.reserve(element_cnt);

	if (unique_elements) { // Generate unique elements.

		if (element_cnt > (1ull << 32)) {
			std::cerr << "Cannot create more than 2^32 unique integers." << std::endl;
			std::exit(1);
		}

		std::unordered_set<uint64_t> set;
		std::size_t c = 0;
		while (c < element_cnt) {
			auto val = rnd();
			if (set.count(val) == 0) {
				data.push_back(val);
				set.insert(val);
				c++;
			}
		}
	} else { // Generate non-unique elements.
		for (std::size_t i = 0; i < element_cnt; i++) {
			data.push_back(rnd());
		}
	}
}

} // namespace internal
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
enum rnd_engine_t {
	RANDOM_DEVICE,
	MERSENNE_TWISTER,
};
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
static void gen_data(std::vector<uint32_t> &data, const std::size_t element_cnt, const rnd_engine_t rnd_engine,
                     const bool unique) {

	std::random_device rnd_device;

	switch (rnd_engine) {
	case RANDOM_DEVICE: {
		auto gen_rand = [&rnd_device]() { return static_cast<uint32_t>(rnd_device()); };
		internal::gen_random_data(data, element_cnt, unique, gen_rand);
		break;
	}
	case MERSENNE_TWISTER: {
		auto gen_rand = [&rnd_device]() {
			std::mt19937 gen(rnd_device());
			std::uniform_int_distribution<uint32_t> dis;
			return dis(gen);
		};
		internal::gen_random_data(data, element_cnt, unique, gen_rand);
		break;
	}
	}
}
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
static void gen_data(std::vector<uint64_t> &data, const std::size_t element_cnt, const rnd_engine_t rnd_engine,
                     const bool unique) {

	std::random_device rnd_device;

	switch (rnd_engine) {
	case RANDOM_DEVICE: {
		auto gen_rand = [&rnd_device]() { return rnd_device() + (static_cast<::std::uint64_t>(rnd_device()) << 32); };
		internal::gen_random_data_64(data, element_cnt, unique, gen_rand);
		break;
	}
	case MERSENNE_TWISTER: {
		std::mt19937_64 gen(rnd_device());
		std::uniform_int_distribution<uint64_t> dis;
		auto gen_rand = [&]() { return dis(gen); };
		internal::gen_random_data_64(data, element_cnt, unique, gen_rand);
		break;
	}
	}
}
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
struct data_generator_64 {

	const rnd_engine_t rnd_engine;
	u1 unique_elements;
	std::unordered_set<uint64_t> set;
	std::function<uint64_t()> rnd_fn;

	data_generator_64(const rnd_engine_t rnd_engine, u1 unique_elements)
	    : rnd_engine(rnd_engine), unique_elements(unique_elements) {

		std::random_device rnd_device;
		std::mt19937_64 gen(rnd_device());
		std::uniform_int_distribution<uint64_t> dis;

		std::function<uint64_t()> rnd = [&rnd_device]() {
			return rnd_device() + (static_cast<::std::uint64_t>(rnd_device()) << 32);
		};

		std::function<uint64_t()> mt = [&]() { return dis(gen); };

		switch (rnd_engine) {
		case RANDOM_DEVICE:
			rnd_fn = rnd;
			break;
		case MERSENNE_TWISTER:
			rnd_fn = mt;
			break;
		}
	}

	uint64_t next() {
		if (unique_elements) { // Generate unique elements.
			while (true) {
				auto val = rnd_fn();
				if (set.count(val) == 0) {
					return val;
				}
			}
		} else { // Generate non-unique elements.
			return rnd_fn();
		}
	}
};
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
struct data_generator_32 {

	u1 unique_elements;
	std::unordered_set<uint32_t> set;

	explicit data_generator_32(u1 unique_elements) : unique_elements(unique_elements){};

	virtual uint32_t next() = 0;
};
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
struct data_generator_32_rnd : data_generator_32 {

	std::random_device rnd_device;

	explicit data_generator_32_rnd(u1 unique_elements) : data_generator_32(unique_elements) {
	}

	uint32_t next() override {
		if (unique_elements) { // Generate unique elements.
			while (true) {
				auto val = rnd_device();
				if (set.count(val) == 0) {
					return val;
				}
			}
		} else { // Generate non-unique elements.
			return rnd_device();
		}
	}
};
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
template <class Container> void set_random_values(Container &container) {
	thread_local std::random_device rd;
	thread_local std::mt19937 engine(rd());

	std::uniform_int_distribution<typename Container::value_type> distribution;
	auto sampler = [&]() { return distribution(engine); };
	std::generate(std::begin(container), std::end(container), sampler);
}
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
struct data_generator_32_mt : data_generator_32 {

	std::random_device rnd_device;
	std::mt19937 gen;
	std::uniform_int_distribution<uint32_t> dis;

	explicit data_generator_32_mt(u1 unique_elements) : data_generator_32(unique_elements), gen(rnd_device()) {
	}

	uint32_t next() override {
		if (unique_elements) { // Generate unique elements.
			while (true) {
				auto val = dis(gen);
				if (set.count(val) == 0) {
					return val;
				}
			}
		} else { // Generate non-unique elements.
			return dis(gen);
		}
	}
};

std::pair<std::string, std::string> split_once(std::string delimited, char delimiter) {
	auto pos = delimited.find_first_of(delimiter);
	return {delimited.substr(0, pos), delimited.substr(pos + 1)};
}
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
struct params_t {

	// Command-line-settable parameters
	std::string kernel_variant    {defaults::kernel_variant};
	bool should_print_results     {defaults::should_print_results};
	int num_gpu_streams           {defaults::num_gpu_streams};
	size_t num_threads_per_block  {defaults::num_threads_per_block};
	int num_blocks                {defaults::num_blocks};
	int num_query_execution_runs  {defaults::num_query_execution_runs};
	std::size_t filter_size       {defaults::filter_size};
	std::size_t probe_size        {defaults::probe_size};
	std::size_t build_size        {defaults::build_size};
	std::size_t gpu_morsel_size   {defaults::gpu_morsel_size};
	std::size_t cpu_morsel_size   {defaults::cpu_morsel_size};

	std::size_t num_repetitions   {defaults::num_repetitions};
	bool gpu  					  {defaults::gpu};
	int cpu_bloomfilter 		  {defaults::cpu_bloomfilter};
	std::size_t selectivity 	  {defaults::selectivity};
	std::string csv_path		  {""};
	bool only_generate {defaults::only_generate};

	std::size_t num_threads 	  {std::thread::hardware_concurrency()};
	std::size_t num_columns 	  {defaults::num_columns};
	std::size_t slowdown 	  	{defaults::slowdown};
	std::size_t num_warmup 		{defaults::num_warmup};
};
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
void print_help(int argc, char** argv) {
    fprintf(stderr, "Unrecognized command line option.\n");
    fprintf(stderr, "Usage: %s [args]\n", argv[0]);
    fprintf(stderr, "   --csv_path=\n");
    fprintf(stderr, "   --filter_size=[default:%u]\n",     defaults::filter_size);
    fprintf(stderr, "   --probe_size=[default:%u]\n",      defaults::probe_size);
    fprintf(stderr, "   --build_size=[default:%u]\n",      defaults::build_size);
    fprintf(stderr, "   --gpu_morsel_size=[default:%u]\n", defaults::gpu_morsel_size);
    fprintf(stderr, "   --cpu_morsel_size=[default:%u]\n", defaults::cpu_morsel_size);
    fprintf(stderr, "   --gpu=[default:%u]\n",     		   defaults::gpu);
    fprintf(stderr, "   --cpu_bloomfilter=[default:%u]\n",     defaults::cpu_bloomfilter);
    fprintf(stderr, "   --selectivity=[default:%u]\n",     defaults::selectivity);
    fprintf(stderr, "   --num_threads=[default:%u]\n",     std::thread::hardware_concurrency());
    fprintf(stderr, "	--only_generate=[default:%u]\n", 	defaults::only_generate);
    fprintf(stderr, "	--slowdown=[default:%u]\n", 	defaults::slowdown);
}
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
params_t parse_command_line(int argc, char **argv) {
	params_t params;

	for (int i = 1; i < argc; i++) {
		auto arg = std::string(argv[i]);
		if (arg.substr(0, 2) != "--") {
			print_help(argc, argv);
			exit(EXIT_FAILURE);
		}
		arg = arg.substr(2);
		auto p = split_once(arg, '=');
		auto &arg_name = p.first;
		auto &arg_value = p.second;
		if(arg_value == "") {
			print_help(argc, argv);
			exit(EXIT_FAILURE);
		} else if (arg_name == "csv_path") {
			params.csv_path = arg_value;
		} else if (arg_name == "num_blocks") {
			params.num_blocks = std::stoll(arg_value);
		} else if (arg_name == "threads-per-block") {
			params.num_threads_per_block = std::stoll(arg_value);
		} else if (arg_name == "filter_size") {
			params.filter_size = std::stoll(arg_value);
		} else if (arg_name == "probe_size") {
			params.probe_size = std::stoll(arg_value);
			if (!params.probe_size) {
				params.probe_size = defaults::probe_size;
			}
		} else if (arg_name == "build_size") {
			params.build_size = std::stoll(arg_value);
			if (!params.build_size) {
				params.build_size = defaults::build_size;
			}
		} else if (arg_name == "slowdown") {
			params.slowdown = std::stoll(arg_value);
		} else if (arg_name == "gpu_morsel_size") {
			params.gpu_morsel_size = std::stoll(arg_value);
		} else if (arg_name == "cpu_morsel_size") {
			params.cpu_morsel_size = std::stoll(arg_value);
		} else if (arg_name == "repetitions") {
			params.num_repetitions = std::stoll(arg_value);
		} else if (arg_name == "selectivity") {
			params.selectivity = std::stoll(arg_value);
		} else if (arg_name == "gpu") {
			params.gpu = std::stoll(arg_value) != 0;
		} else if (arg_name == "only_generate") {
			params.only_generate = std::stoll(arg_value) != 0;
		} else if (arg_name == "cpu_bloomfilter") {
			params.cpu_bloomfilter = std::stoll(arg_value);
		} else if (arg_name == "num_threads") {
			int64_t n = std::stoll(arg_value);
			if (n > 0) {
				params.num_threads = n;
			} else {
				fprintf(stderr, "Invalid num_threads\n");
				print_help(argc, argv);
				exit(EXIT_FAILURE);
			}
		} else {
			print_help(argc, argv);
			exit(EXIT_FAILURE);
		}
	}
	return params;
}
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
template<typename T>
void set_uniform_distributed_values(T* column, size_t range_size) {
    //thread_local allows unique seed for each thread
    thread_local std::random_device rd;     // Will be used to obtain a seed for the random number engine
    thread_local std::mt19937 engine(rd()); //Standard mersenne_twister_engine seeded with rd()

    std::uniform_int_distribution<T> distribution;
    auto sampler = [&]() { return distribution(engine); };   //Use distribution to transform the random unsigned int generated by engine into an int in [0, u32] 
    std::generate(&column[0], column + range_size, sampler); // Initializes the container with random uniform distributed values

	// random shuffle to break sequential access
	//std::shuffle(&column[0], column + range_size, engine);
}
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
template<typename T>
void set_incremental_values(T* column, size_t range_size) {
    //thread_local allows unique seed for each thread
    thread_local std::random_device rd;     // Will be used to obtain a seed for the random number engine
    thread_local std::mt19937 engine(rd()); //Standard mersenne_twister_engine seeded with rd()

    auto increment_one = [n = 0]() mutable { return ++n; };   
    std::generate(&column[0], column + range_size, increment_one); // Initializes the container with random uniform distributed values
}
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
void populate_table(Table& table) {
    for(auto &column : table.columns)
        set_incremental_values((int32_t*)column, table.size());
}
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
void set_selectivity(Table& table_build, Table& table_probe, size_t selectivity) {
	    //thread_local allows unique seed for each thread
    thread_local std::random_device rd;     // Will be used to obtain a seed for the random number engine
    thread_local std::mt19937 engine(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<int32_t> distribution;
	size_t number_of_matches = ((selectivity * table_probe.size()) / 100);
	auto column_build = static_cast<int32_t*>(table_build.columns[0]);
	std::set<int32_t> build_set(&column_build[0], column_build + table_build.size());

    for(auto column_probe : table_probe.columns) {
    	auto column = static_cast<int32_t*>(column_probe);
    	const size_t num = table_probe.size();

    	{
    		// require 'number_of_matches' ... copy build-side multiple times into probe
    		// THIS MIGHT OVERLAP AND MIGHT BE OVERWRITTEN BY THE CONSEQUENT PASS
    		size_t offset = 0;
    		size_t num_copy = (number_of_matches + table_build.size()-1) / table_build.size();
    		assert(table_build.size() > table_probe.size());
    		for (size_t c=0; c<num_copy; c++) {
    			memcpy(&column[offset], column_build, table_build.size() * sizeof(int32_t));

    			offset += table_build.size();
    		}
    	}

    	// replace
   		for(size_t i = number_of_matches; i < num;) {
        	int32_t new_random = distribution(engine);
        	if(build_set.find(new_random) == build_set.end()){
        		column[i] =  new_random;
        		i++;
        	}
        }

        std::shuffle(&column[0], column + num, engine);   
    }
}
//===----------------------------------------------------------------------===//
