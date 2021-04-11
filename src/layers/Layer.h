//
// Created by Sergei Troshin on 18.03.2021.
//

#ifndef NEURAL_NET_CONSTRUCTOR1_LAYER_H
#define NEURAL_NET_CONSTRUCTOR1_LAYER_H

#include <fstream>
#include <vector>
#include "../utils/Parameter.h"
#include "../optimizers/Optimizer.h"

using namespace std;

template<typename T>
class Layer {
public:
    virtual ~Layer() = default;

    virtual Tensor<T> forward(Tensor<T> const & X) = 0;

    virtual Tensor<T> backward(Tensor<T> const & grad) = 0;

    virtual void update(Optimizer & optimizer) = 0;

    virtual void dump(std::ofstream & file) = 0;

    virtual void load(std::ifstream & file) = 0;

    void init_weights() {};

    void get_weights() {};
};

enum LayerType {
    IdentityLayer = 1,
    LinearLayer = 2
};

#endif //NEURAL_NET_CONSTRUCTOR1_LAYER_H
