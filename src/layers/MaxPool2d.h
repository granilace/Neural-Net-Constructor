#pragma once

#include "Layer.h"
#include "utils/Maxpooling.h"
#include <iostream>
#include <vector>

template <typename T> class MaxPool2d : public Layer<T, T, 4, 4> {
  public:
    static constexpr auto layer_type = LayerType::MaxPoolLayer;

    MaxPool2d() {}

    MaxPool2d(size_t kernel_height, size_t kernel_width)
        : kernel_height_(kernel_height),
          kernel_width_(kernel_width) {}

    Tensor<T, 4> forward(Tensor<T, 4> const& X) {
        X_ = X;
        std::pair<Tensor<T, 4>, Tensor<size_t, 5>> output_pair = maxpool2d<T>(X, kernel_height_, kernel_width_);
        Tensor<T, 4> output = output_pair.first;
        indices_ = output_pair.second;
        return output;
    };

    Tensor<T, 4> backward(Tensor<T, 4> const& grad) override {
        Tensor<T, 4> grad_input = grad_maxpool<T>(X_, grad, indices_, kernel_height_, kernel_width_);
        return grad_input;
    };

    void save_weights(std::ofstream& file) override {
    }

    void load_weights(std::ifstream& file) override {
    }

    void update(Optimizer& optimizer) override {
    };

    void init_weights() {
    }

  private:
    size_t kernel_height_;
    size_t kernel_width_;
    Tensor<T, 4> X_;
    Tensor<size_t, 5> indices_;
};