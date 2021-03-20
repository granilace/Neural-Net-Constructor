//
// Created by Sergei Troshin on 18.03.2021.
//

#ifndef NEURAL_NET_CONSTRUCTOR1_IDENTITY_H
#define NEURAL_NET_CONSTRUCTOR1_IDENTITY_H

#include "Layer.h"

template<typename T>
class Identity : public Layer<T> {
public:
    explicit Identity() = default;

    Tensor<T> forward(Tensor<T> const &X) override {
        return X;
    };

    Tensor<T> backward(Tensor<T> const &grad) override {
        return grad;
    };

    void update(Optimizer &optimizer) override {}
};


#endif //NEURAL_NET_CONSTRUCTOR1_IDENTITY_H
