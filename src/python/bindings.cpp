#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "tensor.h"

namespace py = pybind11;
using namespace minitorch;

PYBIND11_MODULE(pyminitorch, m) {
    m.doc() = "MiniTorch: A PyTorch reimplementation in CUDA/C++";
    
    // Device enum
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("CUDA", Device::CUDA);
    
    // DType enum
    py::enum_<DType>(m, "DType")
        .value("FLOAT32", DType::FLOAT32)
        .value("FLOAT64", DType::FLOAT64)
        .value("INT32", DType::INT32)
        .value("INT64", DType::INT64);
    
    // Tensor class
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<const std::vector<int64_t>&, DType, Device>(),
             py::arg("shape"), py::arg("dtype") = DType::FLOAT32, py::arg("device") = Device::CUDA)
        .def(py::init<const std::vector<float>&, const std::vector<int64_t>&, Device>(),
             py::arg("data"), py::arg("shape"), py::arg("device") = Device::CUDA)
        
        // Properties
        .def_property_readonly("shape", &Tensor::shape)
        .def_property_readonly("strides", &Tensor::strides)
        .def_property_readonly("device", &Tensor::device)
        .def_property_readonly("dtype", &Tensor::dtype)
        .def_property_readonly("size", &Tensor::size)
        .def_property_readonly("ndim", &Tensor::ndim)
        .def_property("requires_grad", &Tensor::requires_grad, &Tensor::set_requires_grad)
        .def_property_readonly("grad", &Tensor::grad)
        
        // Operations
        .def("add", &Tensor::add)
        .def("sub", &Tensor::sub)
        .def("mul", &Tensor::mul)
        .def("div", &Tensor::div)
        .def("matmul", &Tensor::matmul)
        
        // Activation functions
        .def("relu", &Tensor::relu)
        .def("sigmoid", &Tensor::sigmoid)
        .def("tanh", &Tensor::tanh)
        
        // Utility operations
        .def("sum", &Tensor::sum, py::arg("dim") = -1)
        .def("mean", &Tensor::mean, py::arg("dim") = -1)
        .def("reshape", &Tensor::reshape)
        .def("transpose", &Tensor::transpose)
        
        // Device operations
        .def("to", &Tensor::to)
        .def("cpu", &Tensor::cpu)
        .def("cuda", &Tensor::cuda)
        
        // Gradient operations
        .def("backward", &Tensor::backward)
        .def("zero_grad", &Tensor::zero_grad)
        
        // Operators
        .def("__add__", &Tensor::add)
        .def("__sub__", &Tensor::sub)
        .def("__mul__", &Tensor::mul)
        .def("__truediv__", &Tensor::div)
        
        // Print
        .def("print", &Tensor::print)
        .def("__repr__", [](const Tensor& t) {
            return "Tensor(shape=" + py::str(py::cast(t.shape())).cast<std::string>() + 
                   ", device=" + (t.device() == Device::CUDA ? "CUDA" : "CPU") + ")";
        });
    
    // Factory functions
    m.def("zeros", &Tensor::zeros, 
          py::arg("shape"), py::arg("dtype") = DType::FLOAT32, py::arg("device") = Device::CUDA);
    m.def("ones", &Tensor::ones, 
          py::arg("shape"), py::arg("dtype") = DType::FLOAT32, py::arg("device") = Device::CUDA);
    m.def("randn", &Tensor::randn, 
          py::arg("shape"), py::arg("dtype") = DType::FLOAT32, py::arg("device") = Device::CUDA);
    m.def("arange", &Tensor::arange, 
          py::arg("start"), py::arg("end"), py::arg("step") = 1.0f, 
          py::arg("dtype") = DType::FLOAT32, py::arg("device") = Device::CUDA);
} 