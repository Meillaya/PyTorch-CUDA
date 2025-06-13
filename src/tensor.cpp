#include "tensor.h"
#include "cuda_kernels.cuh"
#include <stdexcept>
#include <cstring>
#include <random>
#include <algorithm>

namespace minitorch {

Tensor::Tensor(const std::vector<int64_t>& shape, DType dtype, Device device)
    : shape_(shape), device_(device), dtype_(dtype), requires_grad_(false), grad_(nullptr) {
    compute_strides();
    size_ = compute_size();
    allocate_memory();
}

Tensor::Tensor(const std::vector<float>& data, const std::vector<int64_t>& shape, Device device)
    : shape_(shape), device_(device), dtype_(DType::FLOAT32), requires_grad_(false), grad_(nullptr) {
    compute_strides();
    size_ = compute_size();
    
    if (data.size() != size_) {
        throw std::invalid_argument("Data size doesn't match tensor shape");
    }
    
    allocate_memory();
    
    if (device_ == Device::CUDA) {
        cuda::cuda_memcpy_h2d(data_, data.data(), size_ * sizeof(float));
    } else {
        std::memcpy(data_, data.data(), size_ * sizeof(float));
    }
}

Tensor::~Tensor() {
    deallocate_memory();
}

Tensor::Tensor(const Tensor& other)
    : std::enable_shared_from_this<Tensor>(), shape_(other.shape_), strides_(other.strides_), 
      device_(other.device_), dtype_(other.dtype_), size_(other.size_), requires_grad_(other.requires_grad_) {
    allocate_memory();
    
    if (device_ == Device::CUDA) {
        cuda::cuda_memcpy_d2d(data_, other.data_, size_ * dtype_size());
    } else {
        std::memcpy(data_, other.data_, size_ * dtype_size());
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        deallocate_memory();
        shape_ = other.shape_;
        strides_ = other.strides_;
        device_ = other.device_;
        dtype_ = other.dtype_;
        size_ = other.size_;
        requires_grad_ = other.requires_grad_;
        
        allocate_memory();
        
        if (device_ == Device::CUDA) {
            cuda::cuda_memcpy_d2d(data_, other.data_, size_ * dtype_size());
        } else {
            std::memcpy(data_, other.data_, size_ * dtype_size());
        }
    }
    return *this;
}

void Tensor::allocate_memory() {
    size_t bytes = size_ * dtype_size();
    
    if (device_ == Device::CUDA) {
        data_ = cuda::cuda_malloc(bytes);
    } else {
        data_ = std::malloc(bytes);
        if (!data_) {
            throw std::runtime_error("Failed to allocate CPU memory");
        }
    }
}

void Tensor::deallocate_memory() {
    if (data_) {
        if (device_ == Device::CUDA) {
            cuda::cuda_free(data_);
        } else {
            std::free(data_);
        }
        data_ = nullptr;
    }
}

void Tensor::compute_strides() {
    strides_.resize(shape_.size());
    if (shape_.empty()) return;
    
    strides_.back() = 1;
    for (int i = shape_.size() - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

size_t Tensor::compute_size() const {
    size_t size = 1;
    for (auto dim : shape_) {
        size *= dim;
    }
    return size;
}

size_t Tensor::dtype_size() const {
    switch (dtype_) {
        case DType::FLOAT32: return sizeof(float);
        case DType::FLOAT64: return sizeof(double);
        case DType::INT32: return sizeof(int32_t);
        case DType::INT64: return sizeof(int64_t);
        default: return sizeof(float);
    }
}

std::shared_ptr<Tensor> Tensor::add(const std::shared_ptr<Tensor>& other) {
    if (shape_ != other->shape_) {
        throw std::invalid_argument("Tensor shapes must match for addition");
    }
    
    auto result = std::make_shared<Tensor>(shape_, dtype_, device_);
    
    if (device_ == Device::CUDA) {
        if (dtype_ == DType::FLOAT32) {
            cuda::add_wrapper(
                static_cast<const float*>(data_),
                static_cast<const float*>(other->data_),
                static_cast<float*>(result->data_),
                size_
            );
        } else {
            throw std::runtime_error("CUDA operations only support FLOAT32");
        }
    } else {
        // CPU implementation
        const float* a = static_cast<const float*>(data_);
        const float* b = static_cast<const float*>(other->data_);
        float* c = static_cast<float*>(result->data_);
        
        for (size_t i = 0; i < size_; ++i) {
            c[i] = a[i] + b[i];
        }
    }
    
    // Set up autograd
    if (requires_grad_ || other->requires_grad_) {
        result->set_requires_grad(true);
        result->add_input(shared_from_this());
        result->add_input(other);
        
        result->set_grad_fn([](const std::shared_ptr<Tensor>& grad) {
            return std::vector<std::shared_ptr<Tensor>>{grad, grad};
        });
    }
    
    return result;
}

std::shared_ptr<Tensor> Tensor::div(const std::shared_ptr<Tensor>& other) {
    if (shape_ != other->shape_) {
        throw std::invalid_argument("Tensor shapes must match for division");
    }
    
    auto result = std::make_shared<Tensor>(shape_, dtype_, device_);
    
    if (device_ == Device::CUDA) {
        if (dtype_ == DType::FLOAT32) {
            cuda::div_wrapper(
                static_cast<const float*>(data_),
                static_cast<const float*>(other->data_),
                static_cast<float*>(result->data_),
                size_
            );
        } else {
            throw std::runtime_error("CUDA operations only support FLOAT32");
        }
    } else {
        // CPU implementation
        const float* a = static_cast<const float*>(data_);
        const float* b = static_cast<const float*>(other->data_);
        float* c = static_cast<float*>(result->data_);
        
        for (size_t i = 0; i < size_; ++i) {
            c[i] = a[i] / b[i];
        }
    }
    
    return result;
}

std::shared_ptr<Tensor> Tensor::zeros(const std::vector<int64_t>& shape, DType dtype, Device device) {
    auto tensor = std::make_shared<Tensor>(shape, dtype, device);
    
    if (device == Device::CUDA) {
        cuda::fill_wrapper(static_cast<float*>(tensor->data_), 0.0f, tensor->size_);
    } else {
        std::fill(static_cast<float*>(tensor->data_), static_cast<float*>(tensor->data_) + tensor->size_, 0.0f);
    }
    
    return tensor;
}

std::shared_ptr<Tensor> Tensor::ones(const std::vector<int64_t>& shape, DType dtype, Device device) {
    auto tensor = std::make_shared<Tensor>(shape, dtype, device);
    
    if (device == Device::CUDA) {
        cuda::fill_wrapper(static_cast<float*>(tensor->data_), 1.0f, tensor->size_);
    } else {
        std::fill(static_cast<float*>(tensor->data_), static_cast<float*>(tensor->data_) + tensor->size_, 1.0f);
    }
    
    return tensor;
}

std::shared_ptr<Tensor> Tensor::randn(const std::vector<int64_t>& shape, DType dtype, Device device) {
    auto tensor = std::make_shared<Tensor>(shape, dtype, device);
    if (device == Device::CUDA) {
        cuda::randn_wrapper(static_cast<float*>(tensor->data_), tensor->size_);
    } else {
        // Simple CPU randn for demonstration
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> d(0, 1);
        for (size_t i = 0; i < tensor->size_; ++i) {
            static_cast<float*>(tensor->data_)[i] = d(gen);
        }
    }
    return tensor;
}

std::shared_ptr<Tensor> Tensor::sub(const std::shared_ptr<Tensor>& other) {
    if (shape_ != other->shape_) {
        throw std::invalid_argument("Tensor shapes must match for subtraction");
    }
    
    auto result = std::make_shared<Tensor>(shape_, dtype_, device_);
    
    if (device_ == Device::CUDA) {
        if (dtype_ == DType::FLOAT32) {
            cuda::sub_wrapper(
                static_cast<const float*>(data_),
                static_cast<const float*>(other->data_),
                static_cast<float*>(result->data_),
                size_
            );
        } else {
            throw std::runtime_error("CUDA operations only support FLOAT32");
        }
    } else {
        // CPU implementation
        const float* a = static_cast<const float*>(data_);
        const float* b = static_cast<const float*>(other->data_);
        float* c = static_cast<float*>(result->data_);
        
        for (size_t i = 0; i < size_; ++i) {
            c[i] = a[i] - b[i];
        }
    }
    
    return result;
}

std::shared_ptr<Tensor> Tensor::mul(const std::shared_ptr<Tensor>& other) {
    if (shape_ != other->shape_) {
        throw std::invalid_argument("Tensor shapes must match for multiplication");
    }
    
    auto result = std::make_shared<Tensor>(shape_, dtype_, device_);
    
    if (device_ == Device::CUDA) {
        if (dtype_ == DType::FLOAT32) {
            cuda::mul_wrapper(
                static_cast<const float*>(data_),
                static_cast<const float*>(other->data_),
                static_cast<float*>(result->data_),
                size_
            );
        } else {
            throw std::runtime_error("CUDA operations only support FLOAT32");
        }
    } else {
        // CPU implementation
        const float* a = static_cast<const float*>(data_);
        const float* b = static_cast<const float*>(other->data_);
        float* c = static_cast<float*>(result->data_);
        
        for (size_t i = 0; i < size_; ++i) {
            c[i] = a[i] * b[i];
        }
    }
    
    return result;
}

std::shared_ptr<Tensor> Tensor::relu() {
    auto result = std::make_shared<Tensor>(shape_, dtype_, device_);
    
    if (device_ == Device::CUDA) {
        // We'll need to implement this kernel
        throw std::runtime_error("ReLU not yet implemented for CUDA");
    } else {
        // CPU implementation
        const float* input = static_cast<const float*>(data_);
        float* output = static_cast<float*>(result->data_);
        
        for (size_t i = 0; i < size_; ++i) {
            output[i] = std::max(0.0f, input[i]);
        }
    }
    
    return result;
}

std::shared_ptr<Tensor> Tensor::sigmoid() {
    auto result = std::make_shared<Tensor>(shape_, dtype_, device_);
    
    if (device_ == Device::CUDA) {
        // We'll need to implement this kernel
        throw std::runtime_error("Sigmoid not yet implemented for CUDA");
    } else {
        // CPU implementation
        const float* input = static_cast<const float*>(data_);
        float* output = static_cast<float*>(result->data_);
        
        for (size_t i = 0; i < size_; ++i) {
            output[i] = 1.0f / (1.0f + std::exp(-input[i]));
        }
    }
    
    return result;
}

std::shared_ptr<Tensor> Tensor::tanh() {
    auto result = std::make_shared<Tensor>(shape_, dtype_, device_);
    
    if (device_ == Device::CUDA) {
        // We'll need to implement this kernel
        throw std::runtime_error("Tanh not yet implemented for CUDA");
    } else {
        // CPU implementation
        const float* input = static_cast<const float*>(data_);
        float* output = static_cast<float*>(result->data_);
        
        for (size_t i = 0; i < size_; ++i) {
            output[i] = std::tanh(input[i]);
        }
    }
    
    return result;
}

std::shared_ptr<Tensor> Tensor::matmul(const std::shared_ptr<Tensor>& other) {
    // Basic 2D matrix multiplication for now
    if (shape_.size() != 2 || other->shape_.size() != 2) {
        throw std::invalid_argument("Only 2D matrix multiplication supported");
    }
    
    if (shape_[1] != other->shape_[0]) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }
    
    std::vector<int64_t> result_shape = {shape_[0], other->shape_[1]};
    auto result = std::make_shared<Tensor>(result_shape, dtype_, device_);
    
    if (device_ == Device::CUDA) {
        // We'll need to implement CUBLAS integration
        throw std::runtime_error("Matrix multiplication not yet implemented for CUDA");
    } else {
        // Simple CPU implementation
        const float* a = static_cast<const float*>(data_);
        const float* b = static_cast<const float*>(other->data_);
        float* c = static_cast<float*>(result->data_);
        
        int m = shape_[0];
        int n = other->shape_[1];
        int k = shape_[1];
        
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int l = 0; l < k; ++l) {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }
    
    return result;
}

std::shared_ptr<Tensor> Tensor::sum(int dim) {
    // For now, implement sum over all elements
    std::vector<int64_t> result_shape = {1};
    auto result = std::make_shared<Tensor>(result_shape, dtype_, device_);
    
    if (device_ == Device::CUDA) {
        // We'll need to implement sum kernel
        throw std::runtime_error("Sum not yet implemented for CUDA");
    } else {
        // CPU implementation
        const float* input = static_cast<const float*>(data_);
        float* output = static_cast<float*>(result->data_);
        
        float sum = 0.0f;
        for (size_t i = 0; i < size_; ++i) {
            sum += input[i];
        }
        output[0] = sum;
    }
    
    return result;
}

void Tensor::backward() {
    // Basic autograd implementation
    if (!requires_grad_) return;
    
    if (!grad_) {
        grad_ = std::make_shared<Tensor>(shape_, dtype_, device_);
        // Initialize gradient to ones for scalar output
        if (device_ == Device::CUDA) {
            cuda::fill_wrapper(static_cast<float*>(grad_->data_), 1.0f, grad_->size_);
        } else {
            std::fill(static_cast<float*>(grad_->data_), static_cast<float*>(grad_->data_) + grad_->size_, 1.0f);
        }
    }
    
    if (grad_fn_) {
        auto input_grads = grad_fn_(grad_);
        for (size_t i = 0; i < inputs_.size(); ++i) {
            if (inputs_[i]->requires_grad_) {
                if (!inputs_[i]->grad_) {
                    inputs_[i]->grad_ = input_grads[i];
                } else {
                    // Accumulate gradients (add them)
                    inputs_[i]->grad_ = inputs_[i]->grad_->add(input_grads[i]);
                }
            }
        }
    }
}

void Tensor::zero_grad() {
    grad_ = nullptr;
}

std::shared_ptr<Tensor> Tensor::to(Device device) {
    if (device_ == device) {
        return shared_from_this();
    }
    
    auto result = std::make_shared<Tensor>(shape_, dtype_, device);
    
    if (device_ == Device::CPU && device == Device::CUDA) {
        // CPU to CUDA
        cuda::cuda_memcpy_h2d(result->data_, data_, size_ * dtype_size());
    } else if (device_ == Device::CUDA && device == Device::CPU) {
        // CUDA to CPU
        cuda::cuda_memcpy_d2h(result->data_, data_, size_ * dtype_size());
    } else {
        throw std::runtime_error("Unsupported device transfer");
    }
    
    return result;
}

std::shared_ptr<Tensor> Tensor::cpu() {
    return to(Device::CPU);
}

std::shared_ptr<Tensor> Tensor::cuda() {
    return to(Device::CUDA);
}

std::shared_ptr<Tensor> Tensor::arange(float start, float end, float step, DType dtype, Device device) {
    int64_t size = static_cast<int64_t>((end - start) / step);
    std::vector<int64_t> shape = {size};
    auto tensor = std::make_shared<Tensor>(shape, dtype, device);
    
    if (device == Device::CUDA) {
        // For now, create on CPU and copy
        std::vector<float> data(size);
        for (int64_t i = 0; i < size; ++i) {
            data[i] = start + i * step;
        }
        cuda::cuda_memcpy_h2d(tensor->data_, data.data(), size * sizeof(float));
    } else {
        float* data = static_cast<float*>(tensor->data_);
        for (int64_t i = 0; i < size; ++i) {
            data[i] = start + i * step;
        }
    }
    
    return tensor;
}

std::shared_ptr<Tensor> Tensor::reshape(const std::vector<int64_t>& new_shape) {
    size_t new_size = 1;
    for (auto dim : new_shape) {
        new_size *= dim;
    }
    
    if (new_size != size_) {
        throw std::invalid_argument("New shape must have the same number of elements");
    }
    
    auto result = std::make_shared<Tensor>(new_shape, dtype_, device_);
    
    if (device_ == Device::CUDA) {
        cuda::cuda_memcpy_d2d(result->data_, data_, size_ * dtype_size());
    } else {
        std::memcpy(result->data_, data_, size_ * dtype_size());
    }
    
    return result;
}

std::shared_ptr<Tensor> Tensor::transpose(int dim0, int dim1) {
    if (shape_.size() != 2) {
        throw std::invalid_argument("Transpose only supported for 2D tensors");
    }
    
    std::vector<int64_t> new_shape = {shape_[1], shape_[0]};
    auto result = std::make_shared<Tensor>(new_shape, dtype_, device_);
    
    if (device_ == Device::CUDA) {
        throw std::runtime_error("Transpose not yet implemented for CUDA");
    } else {
        // CPU transpose
        const float* src = static_cast<const float*>(data_);
        float* dst = static_cast<float*>(result->data_);
        
        for (int64_t i = 0; i < shape_[0]; ++i) {
            for (int64_t j = 0; j < shape_[1]; ++j) {
                dst[j * shape_[0] + i] = src[i * shape_[1] + j];
            }
        }
    }
    
    return result;
}

std::shared_ptr<Tensor> Tensor::mean(int dim) {
    // Simple implementation: sum / size
    auto sum_result = sum(dim);
    
    if (device_ == Device::CUDA) {
        throw std::runtime_error("Mean not yet implemented for CUDA");
    } else {
        float* data = static_cast<float*>(sum_result->data_);
        data[0] /= static_cast<float>(size_);
    }
    
    return sum_result;
}

void Tensor::print() const {
    if (device_ == Device::CUDA) {
        // Copy to CPU for printing
        std::vector<float> cpu_data(size_);
        cuda::cuda_memcpy_d2h(cpu_data.data(), data_, size_ * sizeof(float));
        
        std::cout << "Tensor(shape=[";
        for (size_t i = 0; i < shape_.size(); ++i) {
            std::cout << shape_[i];
            if (i < shape_.size() - 1) std::cout << ", ";
        }
        std::cout << "], device=CUDA, data=[";
        
        size_t print_size = std::min(size_, size_t(10));
        for (size_t i = 0; i < print_size; ++i) {
            std::cout << cpu_data[i];
            if (i < print_size - 1) std::cout << ", ";
        }
        if (size_ > 10) std::cout << "...";
        std::cout << "])" << std::endl;
    } else {
        const float* data = static_cast<const float*>(data_);
        std::cout << "Tensor(shape=[";
        for (size_t i = 0; i < shape_.size(); ++i) {
            std::cout << shape_[i];
            if (i < shape_.size() - 1) std::cout << ", ";
        }
        std::cout << "], device=CPU, data=[";
        
        size_t print_size = std::min(size_, size_t(10));
        for (size_t i = 0; i < print_size; ++i) {
            std::cout << data[i];
            if (i < print_size - 1) std::cout << ", ";
        }
        if (size_ > 10) std::cout << "...";
        std::cout << "])" << std::endl;
    }
}

} // namespace minitorch 