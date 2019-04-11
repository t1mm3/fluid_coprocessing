cmake_minimum_required(VERSION 3.1)

project(JoinsInWarpSpeed VERSION 0.1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

#subdirectories
include_directories(../dtl/src)
include_directories(../dtl/thirdparty)
include_directories(../cub cub)
include_directories(thirdparty/)
include_directories(src/)
include_directories(src/include)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer ")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer ")


if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(FATAL_ERROR "No build type was specified. Valid choices are either -DCMAKE_BUILD_TYPE=Release or -DCMAKE_BUILD_TYPE=Debug")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

find_package(Threads REQUIRED)

find_package(CUDAAPIWrappers REQUIRED)
include_directories(${CUDA_API_WRAPPERS_INCLUDE_DIRS})
# but what about the library search path?

link_libraries(${CUDA_API_WRAPPERS_LIBS} stdc++fs)

find_package(CUDA 9.1)
if (CUDA_FOUND)
    message("CUDA found")

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -x cu)
    # avoid implicitly synchronization with other streams
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --default-stream per-thread)

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --restrict --std=c++14  -Xcompiler -D__CORRECT_ISO_CPP11_MATH_H_PROTO -D_MWAITXINTRIN_H_INCLUDED)

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --source-in-ptx)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xptxas -dlcm=ca)

    # enable NVCC warnings
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler -Wall)

    # the target architecture(s)  - (see https://en.wikipedia.org/wiki/CUDA#GPUs_supported)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --generate-code arch=compute_52,code=sm_52) # e.g. GTX 970
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --generate-code arch=compute_60,code=sm_60) # e.g. Tesla P100
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --generate-code arch=compute_61,code=sm_61) # e.g. GTX 1080Ti

    # don't show deprecation warnings (which are a lot with CUDA 9)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Wno-deprecated-declarations)

    set(CUDA_NVCC_FLAGS_DEBUG -g --generate-line-info --keep -DDEBUG)
    set(CUDA_NVCC_FLAGS_RELEASE -O3 -g --use_fast_math -DNDEBUG)


    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_SEPARABLE_COMPILATION OFF)

    include_directories(/usr/local/cuda/include)

    set(LIBS ${LIBS} cudart)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_CUDA")


else()
    message("CUDA found")
endif()

### GCC Flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DGPU -DDEBUG -O0 -g -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGPU -O3 -g -march=native")


SET(BLOOM_CUDA_SOURCE_FILES
        ${BLOOM_FILTER_SOURCE_FILES}
        src/bloomfilter/blocked_bloom_benchmark.cu
        )

SET(FRAMEWORK
        ${BLOOM_FILTER_SOURCE_FILES}
        main.cu
        )

SET(BLOOM_SIZE_FILES
        ${BLOOM_FILTER_SOURCE_FILES}
        src/bloomfilter/bloom_size_benchmark.cu
        )

configure_file(build.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/build.hpp)
include_directories(${CMAKE_CURRENT_BINARY_DIR})


#add_library(mainlib SHARED src/task_manager.cpp src/table.cpp src/query.cpp)
#target_compile_options(mainlib PRIVATE -march=native -Wall -pedantic)

#add_library(veclib SHARED src/hash_table.cpp src/vectorized.cpp)
#target_compile_options(veclib PRIVATE -march=native -Wall -pedantic)
#set_target_properties(veclib PROPERTIES COMPILE_FLAGS "-O3 -march=native")


#add_executable(main main.cpp)

#target_link_libraries(main mainlib veclib Threads::Threads)
#target_compile_options(main PRIVATE -march=native -Wall -pedantic)

#benchmark
cuda_add_executable(blocked_bloom_benchmark ${BLOOM_CUDA_SOURCE_FILES})
target_link_libraries(blocked_bloom_benchmark cudart numa pthread)
#framework
cuda_add_executable(main_cu ${FRAMEWORK})
target_link_libraries(main_cu cudart numa pthread)
#bechmark bloom size
cuda_add_executable(bloom_size_benchmark ${BLOOM_SIZE_FILES})
target_link_libraries(blocked_bloom_benchmark cudart numa pthread)
#enable_testing()
#add_executable(test_cpp src/test.cpp src/table.cpp src/query.cpp)
#target_link_libraries(test_cpp mainlib Threads::Threads)
#add_test(NAME test_cpp COMMAND test_cpp)

add_executable(determine_configuration determine_configuration.cpp)
