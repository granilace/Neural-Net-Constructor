//
// Created by Sergei Troshin on 18.03.2021.
//

#ifndef NEURAL_NET_CONSTRUCTOR1_LAYER_H
#define NEURAL_NET_CONSTRUCTOR1_LAYER_H

#endif //NEURAL_NET_CONSTRUCTOR1_LAYER_H

#include "../utils/Parameter.h"
#include "../optimizers/Optimizer.h"

using namespace std;

template<typename T>
class Layer {
    vector<Parameter<T>> parameters;

public:
    explicit Layer();
    ~Layer();

    Eigen::MatrixXd forward(Tensor<T> const & X);

    Eigen::MatrixXd backward(Tensor<T> const & grad);

    void update(Optimizer & optimizer) {
        for (Parameter<T> & parameter : parameters) {
            optimizer.update(parameter);
        }
    }
};
