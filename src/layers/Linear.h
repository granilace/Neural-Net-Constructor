#pragma once

#include <vector>
#include <iostream>
#include "Layer.h"


template<typename T>
class Linear : public Layer<T> {
public:
    Linear(size_t input_size, size_t output_size): weights_(NormTensor<T>(input_size, output_size)), bias_(NormTensor<T>(output_size, 1)) {
    }

    Tensor<T> forward(Tensor<T> const &X) override {
        input_ = X;
        return (X * weights_.tensor).rowwise() + bias_.tensor.rowwise().sum().transpose();
    };

    Tensor<T> backward(Tensor<T> const &grad) override {
        weights_.gradient = input_.transpose() * grad;
        bias_.gradient = grad.colwise().sum().transpose();
        return grad * weights_.tensor.transpose();
    };

    void update(Optimizer &optimizer) override {
        optimizer.update(weights_);
        optimizer.update(bias_);
    };

    void init_weights() {
        weights_.tensor.setOnes();
        bias_.tensor.setOnes();
    }

private:
    Parameter<T> weights_;
    Parameter<T> bias_;
    Tensor<T> input_;
};
