#pragma once
#include <gtest/gtest.h>
#include "utils/Tensor.h"

static const float kEps = 1e-7;

template <typename Type, int NumDums>
void assert_equal(Tensor<Type, NumDums> const & a, Tensor<Type, NumDums> const & b) {
    auto diff = a - b;
    auto diff_squared = diff.unaryExpr([](float d) {return d * d;});
    auto diff_squared_sum = get_value(diff.sum());
    ASSERT_NEAR(diff_squared_sum, 0.0, kEps);
}