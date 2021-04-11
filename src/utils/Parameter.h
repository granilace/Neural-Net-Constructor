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

    Parameter() {}

    explicit Parameter(const Tensor<T> & tensor) : tensor(tensor), gradient(tensor) {
        gradient.setZero();
    }

    void dump(std::ofstream & file) {
        tensor.dump(file);
        gradient.dump(file);
    }

    void load(std::ifstream & file) {
        tensor.load(file);
        gradient.load(file);
    }
};

#endif //NEURAL_NET_CONSTRUCTOR1_PARAMETER_H
