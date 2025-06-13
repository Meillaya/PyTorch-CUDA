#pragma once

#include <vector>
#include <memory>
#include <initializer_list>
#include <iostream>
#include <functional>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace minitorch {

enum class Device {
    CPU,
    CUDA
};

enum class DType {
    FLOAT32,
    FLOAT64,
    INT32,
    INT64
};

class Tensor : public std::enable_shared_from_this<Tensor> {
private:
    void* data_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    Device device_;
    DType dtype_;
    size_t size_;
    bool requires_grad_;
    std::shared_ptr<Tensor> grad_;
    
    // For autograd
    std::vector<std::shared_ptr<Tensor>> inputs_;
    std::function<std::vector<std::shared_ptr<Tensor>>(const std::shared_ptr<Tensor>&)> grad_fn_;

public:
    // Constructors
    Tensor(const std::vector<int64_t>& shape, DType dtype = DType::FLOAT32, Device device = Device::CUDA);
    Tensor(const std::vector<float>& data, const std::vector<int64_t>& shape, Device device = Device::CUDA);
    ~Tensor();
    
    // Copy and move constructors
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    // Properties
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    Device device() const { return device_; }
    DType dtype() const { return dtype_; }
    size_t size() const { return size_; }
    int64_t ndim() const { return shape_.size(); }
    bool requires_grad() const { return requires_grad_; }
    
    // Data access
    void* data() { return data_; }
    const void* data() const { return data_; }
    
    // Gradient operations
    void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }
    std::shared_ptr<Tensor> grad() const { return grad_; }
    void set_grad(std::shared_ptr<Tensor> grad) { grad_ = grad; }
    void backward();
    void zero_grad();
    
    // Basic operations
    std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& other);
    std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor>& other);
    std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor>& other);
    std::shared_ptr<Tensor> div(const std::shared_ptr<Tensor>& other);
    std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& other);
    
    // Activation functions
    std::shared_ptr<Tensor> relu();
    std::shared_ptr<Tensor> sigmoid();
    std::shared_ptr<Tensor> tanh();
    
    // Utility operations
    std::shared_ptr<Tensor> sum(int dim = -1);
    std::shared_ptr<Tensor> mean(int dim = -1);
    std::shared_ptr<Tensor> reshape(const std::vector<int64_t>& new_shape);
    std::shared_ptr<Tensor> transpose(int dim0, int dim1);
    
    // Device operations
    std::shared_ptr<Tensor> to(Device device);
    std::shared_ptr<Tensor> cpu();
    std::shared_ptr<Tensor> cuda();
    
    // Factory methods
    static std::shared_ptr<Tensor> zeros(const std::vector<int64_t>& shape, DType dtype = DType::FLOAT32, Device device = Device::CUDA);
    static std::shared_ptr<Tensor> ones(const std::vector<int64_t>& shape, DType dtype = DType::FLOAT32, Device device = Device::CUDA);
    static std::shared_ptr<Tensor> randn(const std::vector<int64_t>& shape, DType dtype = DType::FLOAT32, Device device = Device::CUDA);
    static std::shared_ptr<Tensor> arange(float start, float end, float step = 1.0f, DType dtype = DType::FLOAT32, Device device = Device::CUDA);
    
    // Operators
    std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& other) { return add(other); }
    std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& other) { return sub(other); }
    std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& other) { return mul(other); }
    std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor>& other) { return div(other); }
    
    // Print
    void print() const;

private:
    void allocate_memory();
    void deallocate_memory();
    void compute_strides();
    size_t compute_size() const;
    size_t dtype_size() const;
    void set_grad_fn(std::function<std::vector<std::shared_ptr<Tensor>>(const std::shared_ptr<Tensor>&)> fn) { grad_fn_ = fn; }
    void add_input(std::shared_ptr<Tensor> input) { inputs_.push_back(input); }
};

} // namespace minitorch 