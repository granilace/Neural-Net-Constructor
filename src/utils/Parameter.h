//
// Created by Sergei Troshin on 18.03.2021.
//

#ifndef NEURAL_NET_CONSTRUCTOR1_PARAMETER_H
#define NEURAL_NET_CONSTRUCTOR1_PARAMETER_H

#include "Tensor.h"

template<typename T>
class Parameter {
    Tensor<T> tensor;
    Tensor<T> gradient; // the same shape as tensor
public:
    explicit Parameter(const Tensor<T> & tensor) : tensor(tensor), gradient(tensor) {
        // zero_like for gradient
    }
};

#endif //NEURAL_NET_CONSTRUCTOR1_PARAMETER_H

