#include "cuda_kernels.cuh"
#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include <cstdio>
#include <ctime>

namespace minitorch {
namespace cuda {

// Element-wise operations
__global__ void add_kernel(const float* a, const float* b, float* out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void sub_kernel(const float* a, const float* b, float* out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] - b[idx];
    }
}

__global__ void mul_kernel(const float* a, const float* b, float* out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void div_kernel(const float* a, const float* b, float* out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] / b[idx];
    }
}

// Activation functions
__global__ void relu_kernel(const float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void relu_backward_kernel(const float* grad_output, const float* input, float* grad_input, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = input[idx] > 0.0f ? grad_output[idx] : 0.0f;
    }
}

__global__ void sigmoid_kernel(const float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void sigmoid_backward_kernel(const float* grad_output, const float* output, float* grad_input, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = grad_output[idx] * output[idx] * (1.0f - output[idx]);
    }
}

__global__ void tanh_kernel(const float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

__global__ void tanh_backward_kernel(const float* grad_output, const float* output, float* grad_input, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = grad_output[idx] * (1.0f - output[idx] * output[idx]);
    }
}

// Initialization kernels
__global__ void fill_kernel(float* data, float value, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

__global__ void setup_curand_kernel(curandState* states, unsigned long seed, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void randn_kernel(float* data, curandState* states, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = curand_normal(&states[idx]);
    }
}

// Reduction operations
__global__ void sum_kernel(const float* input, float* output, size_t size) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < size) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// Memory operations
void cuda_memcpy_h2d(void* dst, const void* src, size_t size) {
    check_cuda_error(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

void cuda_memcpy_d2h(void* dst, const void* src, size_t size) {
    check_cuda_error(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

void cuda_memcpy_d2d(void* dst, const void* src, size_t size) {
    check_cuda_error(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
}

void* cuda_malloc(size_t size) {
    void* ptr;
    check_cuda_error(cudaMalloc(&ptr, size));
    return ptr;
}

void cuda_free(void* ptr) {
    if (ptr) {
        check_cuda_error(cudaFree(ptr));
    }
}

// CUBLAS operations
void cuda_gemm(cublasHandle_t handle, const float* a, const float* b, float* c,
               int m, int n, int k, bool trans_a, bool trans_b) {
    const float alpha = 1.0f, beta = 0.0f;
    
    cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    
    cublasSgemm(handle, op_b, op_a, n, m, k, &alpha, b, trans_b ? k : n, 
                a, trans_a ? m : k, &beta, c, n);
}

// Utility functions
int get_cuda_device_count() {
    int count;
    check_cuda_error(cudaGetDeviceCount(&count));
    return count;
}

void set_cuda_device(int device) {
    check_cuda_error(cudaSetDevice(device));
}

void cuda_synchronize() {
    check_cuda_error(cudaDeviceSynchronize());
}

void add_wrapper(const float* a, const float* b, float* c, size_t size) {
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    add_kernel<<<grid_size, block_size>>>(a, b, c, size);
    check_cuda_error(cudaGetLastError());
}

void sub_wrapper(const float* a, const float* b, float* c, size_t size) {
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    sub_kernel<<<grid_size, block_size>>>(a, b, c, size);
    check_cuda_error(cudaGetLastError());
}

void mul_wrapper(const float* a, const float* b, float* c, size_t size) {
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    mul_kernel<<<grid_size, block_size>>>(a, b, c, size);
    check_cuda_error(cudaGetLastError());
}

void div_wrapper(const float* a, const float* b, float* c, size_t size) {
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    div_kernel<<<grid_size, block_size>>>(a, b, c, size);
    check_cuda_error(cudaGetLastError());
}

void fill_wrapper(float* data, float value, size_t size) {
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    fill_kernel<<<grid_size, block_size>>>(data, value, size);
    check_cuda_error(cudaGetLastError());
}

void randn_wrapper(float* data, size_t size) {
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    curandState* states;
    check_cuda_error(cudaMalloc((void**)&states, size * sizeof(curandState)));
    
    setup_curand_kernel<<<grid_size, block_size>>>(states, time(0), size);
    check_cuda_error(cudaGetLastError());
    
    randn_kernel<<<grid_size, block_size>>>(data, states, size);
    check_cuda_error(cudaGetLastError());
    
    check_cuda_error(cudaFree(states));
}

// Memory management functions
void memcpy_host_to_device(void* dst, const void* src, size_t size) {
    check_cuda_error(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

} // namespace cuda
} // namespace minitorch 