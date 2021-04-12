//
// Created by Sergei Troshin on 18.03.2021.
//

#ifndef NEURAL_NET_CONSTRUCTOR1_IDENTITY_H
#define NEURAL_NET_CONSTRUCTOR1_IDENTITY_H

#include "Layer.h"
#include <fstream>

template <typename T, int InNumDims = 2, int OutNumDims = 2>
class Identity : public Layer<T, T, InNumDims, OutNumDims> {
  public:
    static constexpr auto layer_type = LayerType::IdentityLayer;

    explicit Identity() = default;

    Tensor<T, OutNumDims> forward(Tensor<T, InNumDims> const& X) override {
        return X;
    };

    Tensor<T, InNumDims> backward(Tensor<T, OutNumDims> const& grad) override {
        return grad;
    };

    void save_weights(std::ofstream& file) override {
        auto layer_id = static_cast<char>(LayerType::IdentityLayer);
        file.write(&layer_id, sizeof(layer_id));
    }

    void load_weights(std::ifstream& file) override {
        char layer_id;
        file.read(&layer_id, sizeof(layer_id));
    }

    void update(Optimizer& optimizer) override {}

    void init_weights() {}
};

#endif // NEURAL_NET_CONSTRUCTOR1_IDENTITY_H
