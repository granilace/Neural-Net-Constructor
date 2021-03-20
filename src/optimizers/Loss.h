#pragma once

#include "../utils/Tensor.h"

template<typename T>
T MSE(const Tensor<T>& first, const Tensor<T>& second) {
    return (first - second).array().pow(2).mean();
}

template<typename T>
T LogLoss(const Tensor<T>& probs, const Tensor<T>& labels) {
    return -(labels.array() * probs.array().log() + (1 - labels.array()) * (1 - probs.array()).log()).mean();
}