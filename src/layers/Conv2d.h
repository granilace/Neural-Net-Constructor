#pragma once

#include "Layer.h"
#include "utils/Convolution.h"
#include <iostream>
#include <vector>

template <typename T> class Conv2d : public Layer<T, T, 4, 4> {
  public:
    static constexpr auto layer_type = LayerType::Conv2dLayer;

    Conv2d() {}

    Conv2d(size_t in_channels, size_t out_channels, size_t kernel_height, size_t kernel_width)
        : weights_(NormTensor<T>(in_channels, out_channels, kernel_height, kernel_width)),
          bias_(NormTensor<T>(out_channels)) {}

    Tensor<T, 4> forward(Tensor<T, 4> const& X) {
        X_ = X;
        Tensor<T, 4> output = conv2d<T>(X, weights_.tensor, bias_.tensor);
        return output;
    };

    Tensor<T, 4> backward(Tensor<T, 4> const& grad) override {
        // Tensor<T, 1> bias_dummy();
        Tensor<T, 4> dfdF = grad_weight<T>(X_, grad);
        weights_.gradient += dfdF;
        Tensor<T, 1> dfdb = grad_bias<T>(grad);
        bias_.gradient += dfdb;
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
    Parameter<T, 4> weights_;
    Parameter<T, 1> bias_;
    Tensor<T, 4> input_;
    Tensor<T, 4> X_; // saved for backward
};