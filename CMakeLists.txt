cmake_minimum_required(VERSION 3.1)

project(JoinsInWarpSpeed VERSION 0.1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")


if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(FATAL_ERROR "No build type was specified. Valid choices are either -DCMAKE_BUILD_TYPE=Release or -DCMAKE_BUILD_TYPE=Debug")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

find_package(Threads REQUIRED)

find_package(CUDA 9.1)
if(CUDA_FOUND)
	message("CUDA stuff")
endif()

configure_file(build.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/build.hpp)
include_directories(${CMAKE_CURRENT_BINARY_DIR})


add_library(mainlib STATIC table.cpp query.cpp)
target_compile_options(mainlib PRIVATE -march=native -Wall -pedantic)

add_library(veclib STATIC hash_table.cpp vectorized.cpp)
target_compile_options(veclib PRIVATE -march=native -Wall -pedantic)
set_target_properties(veclib PROPERTIES COMPILE_FLAGS "-O3 -march=native")



add_executable(main main.cpp)

target_link_libraries(main mainlib veclib Threads::Threads)
target_compile_options(main PRIVATE -march=native -Wall -pedantic)

enable_testing()
add_executable(test_cpp test.cpp table.cpp query.cpp hash_table.cpp)
target_link_libraries(test_cpp mainlib veclib Threads::Threads)
add_test(NAME test_cpp COMMAND test_cpp)