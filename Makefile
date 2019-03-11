.PHONY: main clean all format test docs doxygen

all: main

clean:
	@rm -rf cmake-build-release
	@rm -rf cmake-build-debug
	@rm -rf build

bench:
	@DIR=$(pwd)
	@if [ ! -d "../dtl" ]; \
	then \
		cd ..; \
		git clone https://github.com/diegomestre2/bloomfilter-bsd.git dtl; \
		echo "dtl downloaded!"; \
		cd $(DIR); \
	fi
	
	@if [ ! -d "cub" ]; \
	then \
		git clone https://github.com/NVlabs/cub.git; \
	fi

	@if [ ! -d "cuda-api-wrappers/lib" ]; \
	then \
		git clone https://github.com/eyalroz/cuda-api-wrappers; \
		cp -r cuda-api-wrappers/scripts .; \
		cd cuda-api-wrappers; \
		git reset --hard 0f2c8a9a75dece95af1757f55a233e7bf8ddbe5e; \
		cmake .; \
		make; \
		cd; \
		echo "cuda-api-wrappers installed!"; \
	fi

	@mkdir -p build
	@mkdir -p build/release
	@cd build/release && cmake -DCMAKE_BUILD_TYPE=Release ../.. && make blocked_bloom_benchmark


main:
	@DIR=$(pwd)
	@if [ ! -d "../dtl" ]; \
	then \
		cd ..; \
		git clone https://github.com/diegomestre2/bloomfilter-bsd.git dtl; \
		echo "dtl downloaded!"; \
		cd $(DIR); \
	fi
	
	@if [ ! -d "cub" ]; \
	then \
		git clone https://github.com/NVlabs/cub.git; \
	fi

	@if [ ! -d "cuda-api-wrappers/lib" ]; \
	then \
		git clone https://github.com/eyalroz/cuda-api-wrappers; \
		cp -r cuda-api-wrappers/scripts .; \
		cd cuda-api-wrappers; \
		git reset --hard 0f2c8a9a75dece95af1757f55a233e7bf8ddbe5e; \
		cmake .; \
		make; \
		cd; \
		echo "cuda-api-wrappers installed!"; \
	fi

	@mkdir -p build
	@mkdir -p build/release
	@cd build/release && cmake -DCMAKE_BUILD_TYPE=Release ../.. && make main
debug:
	@mkdir -p cmake-build-release
	@mkdir -p cmake-build-release/debug
	@cd cmake-build-release/debug && cmake -DCMAKE_BUILD_TYPE=D