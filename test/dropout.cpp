#include <gtest/gtest.h>

#include "layers/Dropout.h"
#include "test_utils.cpp"


TEST(Dropout, DropoutTestTrain) {
    // size_t in_channels, size_t out_channels, size_t kernel_height, size_t kernel_width

    size_t in_channels = 1;
    size_t out_channels = 1;
    size_t H = 2;
    size_t W = 2;

    
    Tensor<float, 4> input(1, in_channels, H, W);
    input.setConstant({1.f});
    auto dropout = Dropout<float, 4>(0.8);
    dropout.train();
    Tensor<float, 4> expected_output(1, in_channels, H, W);
    expected_output.setValues({{{
       {0, 1},
       {0, 0}
    }}});

    auto output = dropout.forward(input);
    auto grad = output;
    grad.setConstant({1.});
    grad = dropout.backward(grad);
    std::cout << "grad: " << grad << std::endl;
    std::cout << "output: " << output << std::endl;
    
    assert_equal<float, 4>(expected_output, output);
    assert_equal<float, 4>(expected_output, grad);
}

TEST(Dropout, DropoutTestEval) {
    // size_t in_channels, size_t out_channels, size_t kernel_height, size_t kernel_width

    size_t in_channels = 1;
    size_t out_channels = 1;
    size_t H = 2;
    size_t W = 2;

    
    Tensor<float, 4> input(1, in_channels, H, W);
    input.setConstant({1.f});
    float p = 0.8;
    auto dropout = Dropout<float, 4>(p);
    dropout.eval();
    Tensor<float, 4> expected_output(1, in_channels, H, W);
    expected_output.setValues({{{
       {p, p},
       {p, p}
    }}});

    auto output = dropout.forward(input);
    auto grad = output;
    grad.setConstant({1.});
    grad = dropout.backward(grad);
    std::cout << "grad: " << grad << std::endl;
    std::cout << "output: " << output << std::endl;

    assert_equal<float, 4>(expected_output, output);
    assert_equal<float, 4>(expected_output, grad);
}
