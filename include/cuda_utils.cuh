#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

inline void check_cuda_error_fn(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

#define check_cuda_error(err) check_cuda_error_fn(err, __FILE__, __LINE__) 