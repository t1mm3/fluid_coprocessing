cmake_minimum_required(VERSION 3.1)

project(JoinsInWarpSpeed VERSION 0.1)

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

add_executable(main main.cpp)

target_link_libraries(main Threads::Threads)
target_compile_options(main PRIVATE -march=native -Wall -pedantic)