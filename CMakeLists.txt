cmake_minimum_required(VERSION 3.20)
option(KVB_ENABLE_CUDA "Build CUDA kernels and the GPU example" ON)

# Set this before CUDA is enabled; otherwise CMake initializes its own default.
# Respect both -DCMAKE_CUDA_ARCHITECTURES=... and the standard CUDAARCHS env var.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
  set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures for KVBridge")
endif()
project(kvbridge_attention VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(kvbridge_attention_shape STATIC src/attention_shape.cpp)
target_include_directories(kvbridge_attention_shape PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

include(CTest)
if(BUILD_TESTING)
  add_executable(attention_shape_test tests/shape_test.cpp)
  target_link_libraries(attention_shape_test PRIVATE kvbridge_attention_shape)
  add_test(NAME attention_shape_test COMMAND attention_shape_test)
  add_executable(attention_bench_options_test tests/bench_options_test.cpp)
  target_include_directories(attention_bench_options_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples)
  add_test(NAME attention_bench_options_test COMMAND attention_bench_options_test)
  add_executable(attention_reference_test tests/reference_test.cpp)
  target_include_directories(attention_reference_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples)
  target_link_libraries(attention_reference_test PRIVATE kvbridge_attention_shape)
  add_test(NAME attention_reference_test COMMAND attention_reference_test)
  set_tests_properties(attention_shape_test attention_bench_options_test attention_reference_test
    PROPERTIES LABELS "host-only" TIMEOUT 30)
endif()

if(KVB_ENABLE_CUDA)
  enable_language(CUDA)
  add_library(kvbridge_attention STATIC src/attention_shared.cu src/attention_mma.cu)
  target_include_directories(kvbridge_attention PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(kvbridge_attention PUBLIC kvbridge_attention_shape)

  add_executable(attention_bench examples/attention_bench.cu)
  target_link_libraries(attention_bench PRIVATE kvbridge_attention)
endif()
