cmake_minimum_required(VERSION 3.18)

project(bitmax VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_bitmax_cpp MODULE cpp/bitmax/cpu_extension.cpp)
target_link_libraries(_bitmax_cpp PRIVATE Python::NumPy)
target_compile_definitions(_bitmax_cpp PRIVATE VERSION_INFO="${PROJECT_VERSION}")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  target_compile_definitions(_bitmax_cpp PRIVATE BITMAX_ARM64=1)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU|AppleClang")
  target_compile_options(_bitmax_cpp PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS _bitmax_cpp LIBRARY DESTINATION bitmax)

set(BITMAX_BUILD_CUDA_DEFAULT OFF)
if(DEFINED ENV{BITMAX_BUILD_CUDA} AND NOT "$ENV{BITMAX_BUILD_CUDA}" STREQUAL "" AND NOT "$ENV{BITMAX_BUILD_CUDA}" STREQUAL "0")
  set(BITMAX_BUILD_CUDA_DEFAULT ON)
endif()
option(BITMAX_BUILD_CUDA "Build optional CUDA kernels" ${BITMAX_BUILD_CUDA_DEFAULT})

if(BITMAX_BUILD_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  pybind11_add_module(_bitmax_cuda MODULE cpp/bitmax/cuda_extension.cu)
  target_link_libraries(_bitmax_cuda PRIVATE Python::NumPy CUDA::cudart)
  target_compile_definitions(_bitmax_cuda PRIVATE VERSION_INFO="${PROJECT_VERSION}")
  set_target_properties(_bitmax_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  install(TARGETS _bitmax_cuda LIBRARY DESTINATION bitmax)
endif()
