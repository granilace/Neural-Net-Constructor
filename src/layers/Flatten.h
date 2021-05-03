#pragma once

#include "Layer.h"
#include <fstream>

template <typename T>
class Flatten : public Layer<T, T, 4, 2> {
  public:
    static constexpr auto layer_type = LayerType::FlattenLayer;

    Flatten() {}

    Tensor<T, 2> forward(Tensor<T, 4> const& X) override {
        std::cout << "Flatten::forward" << std::endl;
        X_ = X;
        std::cout << "After X_=X" << std::endl;
        std::array<int, 2> out_dimensions;
        out_dimensions[0] = X.dimension(0);
        out_dimensions[1] = 1;
        for (size_t dim = 1; dim < 4; ++dim) {
            out_dimensions[1] *= X.dimension(dim);
        }
        std::cout << "out_dimensions[1]=" << out_dimensions[1] << std::endl;
        return X.reshape(out_dimensions);
    };

    Tensor<T, 4> backward(Tensor<T, 2> const& grad) override {
        return grad.reshape(X_.dimensions());
    };

    void save_weights(std::ofstream& file) override {
        auto layer_id = static_cast<char>(LayerType::FlattenLayer);
        file.write(&layer_id, sizeof(layer_id));
    }

    void load_weights(std::ifstream& file) override {
        char layer_id;
        file.read(&layer_id, sizeof(layer_id));
    }

    void update(Optimizer& optimizer) override {}

    void init_weights() {}

private:
    Tensor<T, 4> X_; // saved for backward
};
