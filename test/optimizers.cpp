#include <gtest/gtest.h>

#include "utils/Tensor.h"
#include "optimizers/Loss.h"

TEST(Loss, MSE) {
    Tensor<float> a(10, 1);
    a.setOnes();
    Tensor<float> b = -a;
    auto mse = MSE(a, b);
    ASSERT_NEAR(mse, 4.0, 1e-7);
}

TEST(Loss, LogLoss) {
    Tensor<float> probs(4, 1);
    probs << 0.1, 0.25, 0.5, 0.9;
    Tensor<float> labels(4, 1);
    labels << 1.0, 0.0, 1.0, 0.0;
    auto log_loss = LogLoss(probs, labels);
    ASSERT_NEAR(log_loss, 1.396499872, 1e-7);
}