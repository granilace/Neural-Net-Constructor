//
// Created by Sergei Troshin on 18.03.2021.
//

#ifndef NEURAL_NET_CONSTRUCTOR1_SEQUENTIAL_H
#define NEURAL_NET_CONSTRUCTOR1_SEQUENTIAL_H

#include <vector>
#include "Layer.h"

template<typename T>
class Sequential : public Layer<T> {
    std::vector<Layer<T>> layers;
public:
    Sequential(std::vector<Layer<T>> & layers);
    ~Sequential();
};


#endif //NEURAL_NET_CONSTRUCTOR1_SEQUENTIAL_H
