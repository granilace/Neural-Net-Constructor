#pragma once

#include "Layer.h"
#include <iostream>
#include <vector>

template <typename T> class Conv2d : public Layer<T, T, 4, 4> {
  public:
    static constexpr auto layer_type = LayerType::Conv2dLayer;

    Conv2d() {}

    Conv2d(size_t in_channels, size_t out_channels, size_t kernel_height, size_t kernel_width)
        : weights_(NormTensor<T>(out_channels, in_channels, kernel_height, kernel_width)),
          bias_(NormTensor<T>(out_channels)) {}

    Tensor<T, 4> forward(Tensor<T, 4> const& X) {
        return X; //XXX implement
    };

    Tensor<T, 4> backward(Tensor<T, 4> const& grad) override {
        return grad; //XXX implement
    };

    void save_weights(std::ofstream& file) override {
        weights_.dump(file);
        bias_.dump(file);
    }

    void load_weights(std::ifstream& file) override {
        weights_.load(file);
        bias_.load(file);
    }

    void update(Optimizer& optimizer) override {
        optimizer.update(weights_);
        optimizer.update(bias_);
    };

    void init_weights() {
        weights_.tensor.setConstant(1);
        bias_.tensor.setConstant(1);
    }

  private:
    Parameter<T, 3> weights_;
    Parameter<T, 1> bias_;
    Tensor<T, 4> input_;
};