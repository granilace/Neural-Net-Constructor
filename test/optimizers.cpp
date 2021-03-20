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