cmake_minimum_required(VERSION 3.18)
project(MiniTorch LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable modern CUDA support
enable_language(CUDA)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
    # Fallback: try to find pybind11 via pip install location
    execute_process(
        COMMAND python -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(PYBIND11_CMAKE_DIR)
        list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
        find_package(pybind11 CONFIG REQUIRED)
    else()
        message(FATAL_ERROR "pybind11 not found. Please install it with: pip install pybind11")
    endif()
endif()

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Compile options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -fno-lto")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-relaxed-constexpr")

# Source files
file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")
file(GLOB_RECURSE CPU_SOURCES "src/*.cpp")
list(FILTER CPU_SOURCES EXCLUDE REGEX ".*/python/.*")
set(SOURCES ${CUDA_SOURCES} ${CPU_SOURCES})

file(GLOB_RECURSE HEADERS "include/*.h" "include/*.cuh")

# Create the main library (static for Python embedding)
add_library(minitorch STATIC ${SOURCES})
target_link_libraries(minitorch PRIVATE CUDA::cudart CUDA::cublas CUDA::curand CUDA::cusparse)
set_property(TARGET minitorch PROPERTY POSITION_INDEPENDENT_CODE ON)

# Python bindings
pybind11_add_module(pyminitorch src/python/bindings.cpp)
target_link_libraries(pyminitorch PRIVATE minitorch CUDA::cudart CUDA::cublas CUDA::curand CUDA::cusparse)
target_compile_definitions(pyminitorch PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

# Set CUDA separable compilation
set_property(TARGET minitorch PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET pyminitorch PROPERTY CUDA_SEPARABLE_COMPILATION ON) 