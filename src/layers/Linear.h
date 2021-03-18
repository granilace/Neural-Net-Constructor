//
// Created by Sergei Troshin on 18.03.2021.
//

#ifndef NEURAL_NET_CONSTRUCTOR1_LINEAR_H
#define NEURAL_NET_CONSTRUCTOR1_LINEAR_H
#include <vector>
#include "Layer.h"

template<typename T>
class Linear : public Layer<T> {
public:
    explicit Linear() {};

    Tensor<T> forward(Tensor<T> const &X) override {};

    Tensor<T> backward(Tensor<T> const &grad) override {};

};

#endif //NEURAL_NET_CONSTRUCTOR1_LINEAR_H
