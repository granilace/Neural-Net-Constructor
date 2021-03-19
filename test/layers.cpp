#include <gtest/gtest.h>

#include "utils/Tensor.h"
#include "utils/Parameter.h"
#include "layers/Identity.h"
#include "layers/Sequential.h"

TEST(Sequential, Identity) {
    Tensor<float> tensor(3, 4);
    tensor.setOnes();

    Tensor<float> vec(4, 1);
    vec.setOnes();

    Parameter<float> parameter(tensor);
    Tensor<float> input = (tensor * vec);

    auto model = Sequential<float>({
           new Identity<float>(),
           new Identity<float>()
    });
    Tensor<float> loss = model.forward(input);
    model.backward(loss);
    ASSERT_NEAR(loss.sum(), 12.1, 1e-7);
}
