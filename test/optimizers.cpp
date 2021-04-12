#include <gtest/gtest.h>

#include "utils/Tensor.h"
#include "optimizers/Loss.h"
#include "layers/Sequential.h"
#include "layers/Linear.h"
#include "optimizers/Optimizer.h"

TEST(Loss, MSE) {
    Tensor<float, 2> a(10, 1);
    a.setConstant(1);
    Tensor<float, 2> b = -a;
    auto mse = MSE(a, b);
    ASSERT_NEAR(mse, 4.0, 1e-7);
}

TEST(Loss, LogLoss) {
    Tensor<float, 2> probs(4, 1);
    probs.setValues({{0.1}, {0.25}, {0.5}, {0.9}});
    Tensor<float, 2> labels(4, 1);
    labels.setValues({{1.0}, {0.0}, {1.0}, {0.0}});
    auto log_loss = LogLoss(probs, labels);
    ASSERT_NEAR(log_loss, 1.396499872, 1e-7);
}

TEST(UpdateStep, TheTest) {
    auto model = Sequential(Linear<float>(2, 10), Linear<float>(10,20));
    Tensor<float, 2> input(5, 2);
    Tensor<float, 2> grad(5, 20);
    input.setConstant(1);
    grad.setConstant(1);
    model.init_weights();
    auto opt = Optimizer(1e-3);
    model.forward(input);
    model.backward(grad);
    model.update(opt);
}
