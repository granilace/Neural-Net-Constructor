#pragma once

#include "layers/Layer.h"
#include "utils/Maxpooling.h"
#include "utils/Tensor.h"
#include <iostream>
#include <vector>
#include <cmath>

template <typename T> class Sigmoid : public Layer<T, T, 2, 2> {
 public:
    static constexpr auto layer_type = LayerType::ActivationLayer;

    Sigmoid() = default;

    Tensor<T, 2> forward(Tensor<T, 2> const& X) {
        X_ = X;
        Tensor<T, 2> output = X;
        for (int i = 0; i != output.size(); ++i) {
            output[i] = sigmoid(output[i]);
        }
        return output;
    };

    Tensor<T, 2> backward(Tensor<T, 4> const& grad) override {
        Tensor<T, 2> grad_input = X_;
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
    Tensor<T, 2> X_;

    static T sigmoid(T inp);
};

template<typename T>
T Sigmoid<T>::sigmoid(T inp) {
    return 1 / (1 + exp(-inp));
}

template <typename T> class ReLU : public Layer<T, T, 2, 2> {
 public:
    static constexpr auto layer_type = LayerType::ActivationLayer;

    ReLU() = default;

    Tensor<T, 2> forward(Tensor<T, 2> const& X) {
        X_ = X;
        Tensor<T, 2> output = X;
        for (int i = 0; i != output.size(); ++i) {
            output[i] = relu(output[i]);
        }
        return output;
    };

    Tensor<T, 2> backward(Tensor<T, 4> const& grad) override {
        Tensor<T, 2> grad_input = X_;
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
    Tensor<T, 2> X_;

    static T relu(T inp);
};

template<typename T>
T ReLU<T>::relu(T inp) {
    return  inp > 0 ? inp : 0;
}

template <typename T> class Softmax : public Layer<T, T, 2, 2> {
public:
    static constexpr auto layer_type = LayerType::ActivationLayer;

    Softmax() = default;

    Tensor<T, 2> forward(Tensor<T, 2> const& X) {
        Tensor<T, 2> output = X.exp();
        for (int b = 0; b < (int)X.dimension(0); ++b) {
            T output_sum = 0;
            for (int i = 0; i < X.dimension(1); ++i) {
                output_sum += X(b, i);
            }
            for (int i = 0; i < (int)X.dimension(1); ++i) {
                output(b, i) /= (output_sum + 1e-9);
            }
        }
        output_ = output;
        return output;
    };

    Tensor<T, 2> backward(Tensor<T, 2> const& grad) override {
        // Tensor<T, 2> grad_input = output_ * (-output_ + 1);
        //return grad_input * grad;
        Tensor<T, 2> ones;
        ones.resize(output_.dimensions());
        ones.setConstant(1);
        return grad * output_ * (ones - output_);
    };

    void save_weights(std::ofstream& file) override {};
    void load_weights(std::ifstream& file) override {};
    void update(Optimizer& optimizer) override {};
    void init_weights() override {};

private:
    Tensor<T, 2> output_;
};