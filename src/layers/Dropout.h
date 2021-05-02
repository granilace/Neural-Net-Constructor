#ifndef NEURAL_NET_CONSTRUCTOR1_IDENTITY_H
#define NEURAL_NET_CONSTRUCTOR1_IDENTITY_H

#include "Layer.h"
#include <fstream>

template <typename T, int NumDims = 2>
class Dropout : public Layer<T, T, NumDims, NumDims> {
  public:
    static constexpr auto layer_type = LayerType::DropoutLayer;

    Dropout() {}
    Dropout(float p = 0.5) : p(p) {}

    Tensor<T, NumDims> forward(Tensor<T, NumDims> const& X) override {
        if (Dropout::is_training) {
            mask = X;
            std::cout << "mask before: " << mask << std::endl;
            mask = tensor_fill_bernoulli<T, NumDims>(mask, p);
            std::cout << "mask: " << mask << std::endl;
            return mask * X;
        } else {
            return X * p;
        }
    };

    Tensor<T, NumDims> backward(Tensor<T, NumDims> const& grad) override {
        if (Dropout::is_training) {
            return grad * mask;
        } else {
            return grad * p;
        }
    };

    void save_weights(std::ofstream& file) override {
        auto layer_id = static_cast<char>(LayerType::DropoutLayer);
        file.write(&layer_id, sizeof(layer_id));
    }

    void load_weights(std::ifstream& file) override {
        char layer_id;
        file.read(&layer_id, sizeof(layer_id));
    }

    void update(Optimizer& optimizer) override {}

    void init_weights() {}

  private:
    float p = 0.5;
    Tensor<T, NumDims> mask;
};

#endif // NEURAL_NET_CONSTRUCTOR1_IDENTITY_H
