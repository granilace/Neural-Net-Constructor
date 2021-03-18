//
// Created by Sergei Troshin on 18.03.2021.
//

#ifndef NEURAL_NET_CONSTRUCTOR1_OPTIMIZER_H
#define NEURAL_NET_CONSTRUCTOR1_OPTIMIZER_H

#include "../utils/Parameter.cpp"


class Optimizer {
public:
    Optimizer() {

    }

    template<typename T>
    void update(Parameter<T> & parameter) {

    };
};


#endif //NEURAL_NET_CONSTRUCTOR1_OPTIMIZER_H
