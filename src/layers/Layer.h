//
// Created by Sergei Troshin on 18.03.2021.
//

#ifndef NEURAL_NET_CONSTRUCTOR1_LAYER_H
#define NEURAL_NET_CONSTRUCTOR1_LAYER_H

#include <vector>
#include "../utils/Parameter.h"
#include "../optimizers/Optimizer.h"

using namespace std;

template<typename T>
class Layer {
    vector<Parameter<T>> parameters;

public:
    virtual ~Layer() = default;

    virtual Tensor<T> forward(Tensor<T> const & X) = 0;

    virtual Tensor<T> backward(Tensor<T> const & grad) = 0;

    void update(Optimizer & optimizer) {
        for (Parameter<T> & parameter : parameters) {
            optimizer.update(parameter);
        }
    }
};

#endif //NEURAL_NET_CONSTRUCTOR1_LAYER_H


