# MiniTorch: PyTorch from Scratch in CUDA/C++

A educational reimplementation of PyTorch's core functionality using pure CUDA and C++. This project aims to provide deep insights into PyTorch's internal workings by building tensor operations, automatic differentiation, and neural network layers from the ground up.

## Project Goals

- **Educational**: Understand PyTorch internals through hands-on implementation
- **Performance**: Leverage CUDA for GPU-accelerated tensor operations  
- **Completeness**: Implement core features including autograd, neural networks, and optimizers
- **Clean Architecture**: Well-structured, readable C++/CUDA codebase

## Architecture

```
MiniTorch/
├── include/           # Header files
│   ├── tensor.h       # Core tensor class
│   ├── cuda_kernels.cuh # CUDA kernel declarations
│   └── nn.h           # Neural network modules
├── src/               # Implementation files
│   ├── tensor.cpp     # Tensor operations (CPU fallbacks)
│   ├── cuda_kernels.cu # CUDA kernel implementations
│   └── python/        # Python bindings
│       └── bindings.cpp
├── examples/          # Usage examples
└── tests/             # Unit tests
```

## Key Features

### Tensor Operations
- **Multi-device support**: CPU and CUDA tensors
- **Memory management**: Efficient GPU memory allocation
- **Broadcasting**: Element-wise operations with shape broadcasting
- **Basic ops**: Add, subtract, multiply, divide, matrix multiplication

### Automatic Differentiation
- **Reverse-mode AD**: Gradient computation via backpropagation
- **Gradient accumulation**: Support for multiple backward passes
- **Dynamic computation graphs**: PyTorch-style autograd system

### Neural Networks
- **Layers**: Linear, ReLU, Sigmoid activation functions
- **Loss functions**: MSE, Cross-entropy loss
- **Optimizers**: SGD, Adam with momentum and weight decay

## Quick Start

### Prerequisites
- CUDA Toolkit (11.0+)
- CMake (3.18+)
- Python 3.8+
- pybind11

### Building

```bash
# Clone the repository
git clone <your-repo-url>
cd PyTorch-CUDA

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make -j$(nproc)

# Install Python package (using uv)
cd ..
uv pip install -e .
```

### Usage Example

```python
import pyminitorch as mt

# Create tensors
a = mt.ones([3, 4])
b = mt.randn([3, 4])

# Enable gradients
a.requires_grad = True
b.requires_grad = True

# Forward pass
c = a + b
loss = c.sum()

# Backward pass
loss.backward()

# Access gradients
print("Gradient of a:", a.grad)
```

## Learning Path

1. **Tensor Basics**: Start with `examples/simple_example.py`
2. **Neural Networks**: Explore `examples/neural_network.py`
3. **CUDA Kernels**: Study `src/cuda_kernels.cu` for GPU implementations
4. **Autograd System**: Examine gradient computation in `src/tensor.cpp`

## Testing

```bash
# Run C++ tests
cd build && make test

# Run Python tests
python -m pytest tests/
```

## Contributing

This is an educational project - contributions and improvements are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch Internals](http://blog.ezyang.com/2019/05/pytorch-internals/)
- [Automatic Differentiation](https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation)

