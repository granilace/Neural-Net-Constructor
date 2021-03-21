//
// Created by Sergei Troshin on 18.03.2021.
//

#ifndef NEURAL_NET_CONSTRUCTOR1_PARAMETER_H
#define NEURAL_NET_CONSTRUCTOR1_PARAMETER_H

#include "Tensor.h"

template<typename T>
class Parameter {
public:
    Tensor<T> tensor;
    Tensor<T> gradient; // the same shape as tensor

    explicit Parameter(const Tensor<T> & tensor) : tensor(tensor), gradient(tensor) {
        gradient.setZero();
    }
};

#endif //NEURAL_NET_CONSTRUCTOR1_PARAMETER_H
