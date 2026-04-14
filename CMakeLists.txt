cmake_minimum_required(VERSION 3.18)
project(transformer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# GPU architecture: auto-detected by cmake, or override with
#   cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

# main.cpp includes transformer_naive.cu, so treat the whole unit as CUDA
set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE CUDA)

add_executable(transformer_naive src/main.cpp)
target_include_directories(transformer_naive PRIVATE src/)
target_compile_options(transformer_naive PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O2>)
