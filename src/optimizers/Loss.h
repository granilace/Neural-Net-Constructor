#pragma once

#include "../utils/Tensor.h"

template<typename T, int NumDims>
T MSE(const Tensor<T, NumDims>& first, const Tensor<T, NumDims>& second) {
    return get_value((first - second).square().mean());
}

template<typename T, int NumDims>
T LogLoss(const Tensor<T, NumDims>& probs, const Tensor<T, NumDims>& labels) {
    return get_value(-(labels * probs.log() + (1 - labels) * (1 - probs).log()).mean());
}