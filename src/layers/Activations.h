#pragma once

#include "Layer.h"
#include "utils/Maxpooling.h"
#include "Tensor.h"
#include <iostream>
#include <vector>
#include <cmath>

template <typename T, typename InNumDims> class Sigmoid : public Layer<T, T, InNumDims, InNumDims> {
 public:
    static constexpr auto layer_type = LayerType::ActivationLayer;

    Sigmoid() = default;

    Tensor<T, InNumDims> forward(Tensor<T, InNumDims> const& X) {
        X_ = X;
        Tensor<T, InNumDims> output = X;
        for (int i = 0; i != output.size(); ++i) {
            output[i] = sigmoid(output[i]);
        }
        return output;
    };

    Tensor<T, InNumDims> backward(Tensor<T, 4> const& grad) override {
        Tensor<T, InNumDims> grad_input = X_;
        for (int i = 0; i != grad_input.size(); ++i) {
            grad_input[i] = sigmoid(grad_input[i]) * (1 - sigmoid(grad_input[i]));
        }
        return grad_input * grad;
    };

    void save_weights(std::ofstream& file) override {};
    void load_weights(std::ifstream& file) override {};
    void update(Optimizer& optimizer) override {};
    void init_weights() override {};

 private:
    Tensor<T, InNumDims> X_;

    static T sigmoid(T inp);
};

template<typename T>
T Sigmoid::sigmoid(T inp) {
    return 1 / (1 + exp(-x));
}

template <typename T, typename InNumDims> class ReLU : public Layer<T, T, InNumDims, InNumDims> {
 public:
    static constexpr auto layer_type = LayerType::ActivationLayer;

    ReLU() = default;

    Tensor<T, InNumDims> forward(Tensor<T, InNumDims> const& X) {
        X_ = X;
        Tensor<T, InNumDims> output = X;
        for (int i = 0; i != output.size(); ++i) {
            output[i] = relu(output[i]);
        }
        return output;
    };

    Tensor<T, InNumDims> backward(Tensor<T, 4> const& grad) override {
        Tensor<T, InNumDims> grad_input = X_;
        for (int i = 0; i != grad_input.size(); ++i) {
            grad_input[i] = grad_input[i] > 0 ? 1 : 0;
        }
        return grad_input * grad;
    };

    void save_weights(std::ofstream& file) override {};
    void load_weights(std::ifstream& file) override {};
    void update(Optimizer& optimizer) override {};
    void init_weights() override {};

 private:
    Tensor<T, InNumDims> X_;

    static T relu(T inp);
};

template<typename T>
T ReLU::relu(T inp) {
    return  inp > 0 ? inp : 0;
}