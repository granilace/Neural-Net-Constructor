//
// Created by Sergei Troshin on 18.03.2021.
//

#ifndef NEURAL_NET_CONSTRUCTOR1_SEQUENTIAL_H
#define NEURAL_NET_CONSTRUCTOR1_SEQUENTIAL_H

#include <vector>
#include <memory>
#include <initializer_list>
#include "Identity.h"
#include "Linear.h"
#include "Layer.h"

template<typename T>
class Sequential : public Layer<T> {
    std::vector<std::shared_ptr<Layer<T>>> layers;
public:
    Sequential() {}

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

    void dump(std::ofstream & file) override {
        for (auto& layer: this->layers) {
            char layer_id;
            if (dynamic_cast<Identity<T>*>(layer.get())) {
                layer_id = LayerType::IdentityLayer;
            } else if (dynamic_cast<Linear<T>*>(layer.get())) {
                layer_id = LayerType::LinearLayer;
            } else {
                throw std::runtime_error("Unsupported dump layer");
            }
            file.write(&layer_id, sizeof(layer_id));
            layer->dump(file);
        }
    }

    void load(std::ifstream & file) override {
        this->layers = {};
        while (file.peek() != EOF) {
            char layer_id;
            file.read(&layer_id, sizeof(layer_id));
            if (layer_id == LayerType::IdentityLayer) {
                auto layer = new Identity<T>();
                layer->load(file);
                this->layers.emplace_back(layer);
            } else if (layer_id == LayerType::LinearLayer) {
                auto layer = new Linear<T>();
                layer->load(file);
                this->layers.emplace_back(layer);
            } else {
                throw std::runtime_error("Unsupported load layer");
            }
        }
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
