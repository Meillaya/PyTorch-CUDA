#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include "cuda_utils.cuh"

namespace minitorch {
namespace cuda {

// Element-wise operations
__global__ void add_kernel(const float* a, const float* b, float* out, size_t size);
__global__ void sub_kernel(const float* a, const float* b, float* out, size_t size);
__global__ void mul_kernel(const float* a, const float* b, float* out, size_t size);
__global__ void div_kernel(const float* a, const float* b, float* out, size_t size);

// Activation functions
__global__ void relu_kernel(const float* input, float* output, size_t size);
__global__ void relu_backward_kernel(const float* grad_output, const float* input, float* grad_input, size_t size);
__global__ void sigmoid_kernel(const float* input, float* output, size_t size);
__global__ void sigmoid_backward_kernel(const float* grad_output, const float* output, float* grad_input, size_t size);
__global__ void tanh_kernel(const float* input, float* output, size_t size);
__global__ void tanh_backward_kernel(const float* grad_output, const float* output, float* grad_input, size_t size);

// Reduction operations
__global__ void sum_kernel(const float* input, float* output, size_t size);
__global__ void mean_kernel(const float* input, float* output, size_t size);

// Initialization kernels
__global__ void fill_kernel(float* data, float value, size_t size);
__global__ void randn_kernel(float* data, curandState* states, size_t size);
__global__ void setup_curand_kernel(curandState* states, unsigned long seed, size_t size);

// Memory operations
void cuda_memcpy_h2d(void* dst, const void* src, size_t size);
void cuda_memcpy_d2h(void* dst, const void* src, size_t size);
void cuda_memcpy_d2d(void* dst, const void* src, size_t size);
void* cuda_malloc(size_t size);
void cuda_free(void* ptr);

// CUBLAS operations
void cuda_gemm(cublasHandle_t handle, const float* a, const float* b, float* c,
               int m, int n, int k, bool trans_a = false, bool trans_b = false);

// Utility functions
int get_cuda_device_count();
void set_cuda_device(int device);
void cuda_synchronize();

// Thread block configurations
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

inline int get_num_blocks(size_t size, int block_size = BLOCK_SIZE) {
    return (size + block_size - 1) / block_size;
}

// Wrapper functions for tensor operations
void add_wrapper(const float* a, const float* b, float* c, size_t size);
void sub_wrapper(const float* a, const float* b, float* c, size_t size);
void mul_wrapper(const float* a, const float* b, float* c, size_t size);
void div_wrapper(const float* a, const float* b, float* c, size_t size);
void fill_wrapper(float* data, float value, size_t size);
void randn_wrapper(float* data, size_t size);

} // namespace cuda
} // namespace minitorch 