//
// Created by Sergei Troshin on 18.03.2021.
//
#pragma once

#include <vector>
#include "../utils/Parameter.cpp"


class Optimizer {
private:
    float learning_rate;
public:
    // On init just pass vector with all parameters
    Optimizer(float learning_rate_) : learning_rate(learning_rate_) {}

    template <typename T, int NumDims>
    void update(Parameter<T, NumDims>& p) {
        p.tensor -= learning_rate * p.gradient;
    };
};