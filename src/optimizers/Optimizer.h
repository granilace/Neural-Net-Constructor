//
// Created by Sergei Troshin on 18.03.2021.
//

#ifndef NEURAL_NET_CONSTRUCTOR1_OPTIMIZER_H
#define NEURAL_NET_CONSTRUCTOR1_OPTIMIZER_H

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


#endif //NEURAL_NET_CONSTRUCTOR1_OPTIMIZER_H
