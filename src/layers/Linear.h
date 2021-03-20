#pragma once

#include <vector>
#include "Layer.h"

template<typename T>
class Linear : public Layer<T> {
public:
    Linear(size_t input_size, size_t output_size): weights_(NormTensor<T>(input_size, output_size)), bias_(NormTensor<T>(output_size, 1)) {
    }

    Tensor<T> forward(Tensor<T> const &X) override {
        return (X * weights_).rowwise() + bias_.rowwise().sum().transpose();
    };

    Tensor<T> backward(Tensor<T> const &grad) override {
        return grad * weights_.transpose();
    };

private:
    Tensor<T> weights_;
    Tensor<T> bias_;
};

