#pragma once

#include "tensor.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>

namespace minitorch {
namespace nn {

class Module {
public:
    virtual ~Module() = default;
    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) = 0;
    virtual std::vector<std::shared_ptr<Tensor>> parameters() = 0;
    virtual void train(bool mode = true) { training_ = mode; }
    virtual void eval() { train(false); }
    bool is_training() const { return training_; }

protected:
    bool training_ = true;
};

class Linear : public Module {
private:
    std::shared_ptr<Tensor> weight_;
    std::shared_ptr<Tensor> bias_;
    int in_features_;
    int out_features_;
    bool use_bias_;

public:
    Linear(int in_features, int out_features, bool bias = true, Device device = Device::CUDA);
    
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    
    std::shared_ptr<Tensor> weight() const { return weight_; }
    std::shared_ptr<Tensor> bias() const { return bias_; }
};

class ReLU : public Module {
public:
    ReLU() = default;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override { return {}; }
};

class Sigmoid : public Module {
public:
    Sigmoid() = default;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override { return {}; }
};

class Sequential : public Module {
private:
    std::vector<std::shared_ptr<Module>> modules_;

public:
    Sequential() = default;
    Sequential(std::initializer_list<std::shared_ptr<Module>> modules);
    
    void add_module(std::shared_ptr<Module> module);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    
    void train(bool mode = true) override;
    void eval() override;
};

// Loss functions
class MSELoss {
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);
};

class CrossEntropyLoss {
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);
};

// Optimizers
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    virtual void zero_grad() = 0;
    
protected:
    std::vector<std::shared_ptr<Tensor>> parameters_;
};

class SGD : public Optimizer {
private:
    float lr_;
    float momentum_;
    float weight_decay_;
    std::vector<std::shared_ptr<Tensor>> momentum_buffers_;

public:
    SGD(const std::vector<std::shared_ptr<Tensor>>& parameters, 
        float lr, float momentum = 0.0f, float weight_decay = 0.0f);
    
    void step() override;
    void zero_grad() override;
};

class Adam : public Optimizer {
private:
    float lr_;
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;
    std::vector<std::shared_ptr<Tensor>> m_buffers_;
    std::vector<std::shared_ptr<Tensor>> v_buffers_;
    int step_count_;

public:
    Adam(const std::vector<std::shared_ptr<Tensor>>& parameters,
         float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, 
         float eps = 1e-8f, float weight_decay = 0.0f);
    
    void step() override;
    void zero_grad() override;
};

} // namespace nn
} // namespace minitorch 