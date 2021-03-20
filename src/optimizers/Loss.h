#pragma once

#include "../utils/Tensor.h"

template<typename T>
T MSE(const Tensor<T>& first, const Tensor<T>& second) {
    Tensor<float> diff = first - second;
    Tensor<float> diff_squared = diff.unaryExpr([](T value) {return value * value;});
    return diff_squared.mean();
}
