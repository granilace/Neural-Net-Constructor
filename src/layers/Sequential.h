//
// Created by Sergei Troshin on 18.03.2021.
//

#ifndef NEURAL_NET_CONSTRUCTOR1_SEQUENTIAL_H
#define NEURAL_NET_CONSTRUCTOR1_SEQUENTIAL_H

#include <vector>
#include <initializer_list>
#include "Layer.h"

template<typename T>
class Sequential : public Layer<T> {
    std::vector<std::unique_ptr<Layer<T>>> layers;
public:
//    explicit Sequential(std::vector<std::unique_ptr<Layer<T>>> layers) : layers(layers) {}
    Sequential(std::initializer_list<Layer<T> *> layers)  {
        for (Layer<T> * it : layers) {
            this->layers.emplace_back(it);
        }
    }

    Tensor<T> forward(Tensor<T> const & X) override {
        Tensor<T> t = X;
        for (auto & layer : layers) {
            t = layer->forward(t);
        }
        return t;
    }

    Tensor<T> backward(Tensor<T> const & grad) override {
        Tensor<T> t = grad;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            auto & layer = *it;
            t = layer->backward(t);
        }
        return t;
    }

    void update(Optimizer &optimizer) override {
        for (const auto &layer : layers) {
            layer->update(optimizer);
        }
    }

    void init_weights() {
        for (const auto &layer : layers) {
            layer->init_weights();
        }
    }
};

#endif //NEURAL_NET_CONSTRUCTOR1_SEQUENTIAL_H
