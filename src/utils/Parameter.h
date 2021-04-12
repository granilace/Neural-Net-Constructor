//
// Created by Sergei Troshin on 18.03.2021.
//

#ifndef NEURAL_NET_CONSTRUCTOR1_PARAMETER_H
#define NEURAL_NET_CONSTRUCTOR1_PARAMETER_H

#include "Tensor.h"

template <typename T, int NumDims = 2> class Parameter {
  public:
    Tensor<T, NumDims> tensor;
    Tensor<T, NumDims> gradient; // the same shape as tensor

    Parameter() {}

    explicit Parameter(const Tensor<T, NumDims>& tensor)
        : tensor(tensor), gradient(tensor) {
        gradient.setZero();
    }

    void dump(std::ofstream& file) {
        save_weights(tensor, file);
        save_weights(gradient, file);
    }

    void load(std::ifstream& file) {
        load_weights(tensor, file);
        load_weights(gradient, file);
    }
};

template <typename T> class ParameterMatrix : public Parameter<T, 2> {
  public:
    MatrixType<T> tensor;
    MatrixType<T> gradient; // the same shape as tensor

    ParameterMatrix() {}

    explicit ParameterMatrix(const MatrixType<T>& tensor)
        : tensor(tensor), gradient(tensor) {
        gradient.setZero();
    }

    void dump(std::ofstream& file) {
        save_weights(tensor, file);
        save_weights(gradient, file);
    }

    void load(std::ifstream& file) {
        load_weights(tensor, file);
        load_weights(gradient, file);
    }
};

#endif // NEURAL_NET_CONSTRUCTOR1_PARAMETER_H
