#pragma once

#include "Layer.h"
#include <iostream>
#include <vector>

template <typename T> class Linear : public Layer<T, T, 2, 2> {
  public:
    static constexpr auto layer_type = LayerType::LinearLayer;

    Linear() {}

    Linear(size_t input_size, size_t output_size)
        : weights_(Tensor_to_Matrix(NormTensor<T>(input_size, output_size))),
          bias_(Tensor_to_Matrix(NormTensor<T>(output_size, 1))) {}

    Tensor<T, 2> forward(Tensor<T, 2> const& X) {
        input_ = Tensor_to_Matrix(X);
        MatrixType<T> res = (input_ * weights_.tensor).rowwise() +
                            bias_.tensor.rowwise().sum().transpose();
        return Matrix_to_Tensor(res);
    };

    Tensor<T, 2> backward(Tensor<T, 2> const& grad) override {
        MatrixType<T> m_grad = Tensor_to_Matrix(grad);
        weights_.gradient = input_.transpose() * m_grad;
        bias_.gradient = m_grad.colwise().sum().transpose();
        MatrixType<T> new_grad = m_grad * weights_.tensor.transpose();
        return Matrix_to_Tensor(new_grad);
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
    ParameterMatrix<T> weights_;
    ParameterMatrix<T> bias_;
    MatrixType<T> input_;
};
