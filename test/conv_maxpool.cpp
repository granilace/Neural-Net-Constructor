#include <gtest/gtest.h>

#include "layers/Conv2d.h"
#include "layers/MaxPool2d.h"
#include "utils/Convolution.h"
#include "test_utils.cpp"


TEST(Conv2dTest, Basic) {
    // size_t in_channels, size_t out_channels, size_t kernel_height, size_t kernel_width

    size_t in_channels = 1;
    size_t out_channels = 1;
    size_t kernel_height = 2;
    size_t kernel_width = 2;
    size_t H = 2;
    size_t W = 2;

    
    Tensor<float, 4> input(1, in_channels, H, W);
    Tensor<float, 4> kernel(in_channels, out_channels, kernel_height, kernel_width);
    Tensor<float, 1> bias(out_channels);
    Tensor<float, 4> output;
    output.setZero();
    input.setValues({{{
        {1, 2},
        {3, 4}
    }}}); // batch size x h x w x in_channels
    kernel.setValues({{{
        {1, 2},
        {3, 4}
    }}});  
    bias.setValues({1});

    // Act
    output = conv2d<float>(input, kernel, bias);
    ASSERT_NEAR(output(0), 31, kEps);
    
    auto conv = Conv2d<float>(kernel, bias); 
    output = conv.forward(input);
    ASSERT_NEAR(output(0), 31, kEps);

    // MaxPool2d<float> mp = MaxPool2d<float>(2, 2);
    // output = mp.forward(input);
    
    
}

TEST(Conv2dTest, TestHW) {
    // size_t in_channels, size_t out_channels, size_t kernel_height, size_t kernel_width

    size_t in_channels = 1;
    size_t out_channels = 2;
    size_t kernel_height = 2;
    size_t kernel_width = 2;
    size_t H = 3;
    size_t W = 3;

    // auto conv = Conv2d<float>(in_channels, out_channels, kernel_height, kernel_width); 
    Tensor<float, 4> input(1, in_channels, H, W);
    Tensor<float, 4> kernel(in_channels, out_channels, kernel_height, kernel_width);
    Tensor<float, 1> bias(out_channels);
    Tensor<float, 4> output;
    output.setZero();
    input.setValues({{{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    }}}); // batch size x h x w x in_channels
    kernel.setValues({{
        {
            {1, 2},
            {3, 4}
        },
        {
            {-1, -2},
            {-3, -4}
        },
    }});  
    bias.setValues({0});

    // Act
    output = conv2d<float>(input, kernel, bias);
    // std::cout << "output" << output;
    Tensor<float, 0> out = output.sum();
    // std::cout << out(0);
    ASSERT_NEAR(out(0), 0, kEps);
}


TEST(Conv2dTest, TestBackward) {
    // size_t in_channels, size_t out_channels, size_t kernel_height, size_t kernel_width

    size_t in_channels = 1;
    size_t out_channels = 1;
    size_t kernel_height = 2;
    size_t kernel_width = 2;
    size_t H = 3;
    size_t W = 3;

    // auto conv = Conv2d<float>(in_channels, out_channels, kernel_height, kernel_width); 
    Tensor<float, 4> input(1, in_channels, H, W);
    Tensor<float, 4> kernel(in_channels, out_channels, kernel_height, kernel_width);
    Tensor<float, 1> bias(out_channels);
    Tensor<float, 4> output;
    output.setZero();
    input.setValues({{{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    }}}); // batch size x h x w x in_channels
    kernel.setValues({{
        {
            {1, 2},
            {3, 4}
        }
    }});  
    bias.setValues({0});

    // Act
    auto conv = Conv2d<float>(kernel, bias); 
    output = conv.forward(input);
    Tensor<float, 4> grad_out = output;
    grad_out.setConstant(1);
    auto grad_input = conv.backward(grad_out);

    // std::cout << "output " << output << std::endl; ;
    // std::cout << "grad_input " << grad_input << std::endl; 
    // std::cout << "grad_weight " << conv.get_weights().gradient << std::endl;
    // std::cout << out(0);


    Tensor<float, 0> out = conv.get_weights().gradient.sum();
    ASSERT_NEAR(out(0), 80, kEps);

    Tensor<float, 4> grad_input_expected(1, in_channels, H, W);
    grad_input_expected.setValues({{{
        {1.,  3.,  2.},
        {4., 10.,  6.},
        {3.,  7.,  4.}
    }}}); 
    assert_equal(grad_input_expected, grad_input);
}

TEST(Conv2dTest, TestMaxpool2d) {
    // size_t in_channels, size_t out_channels, size_t kernel_height, size_t kernel_width

    size_t in_channels = 1;
    size_t out_channels = 1;
    size_t kernel_height = 2;
    size_t kernel_width = 2;
    size_t H = 3;
    size_t W = 3;

    // auto conv = Conv2d<float>(in_channels, out_channels, kernel_height, kernel_width); 
    Tensor<float, 4> input(1, in_channels, H, W);
    Tensor<float, 4> kernel(in_channels, out_channels, kernel_height, kernel_width);
    Tensor<float, 1> bias(out_channels);
    Tensor<float, 4> output;
    output.setZero();
    input.setValues({{{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    }}}); // batch size x h x w x in_channels
    
    // Expected output
    Tensor<float, 4> exp_output(1, in_channels, 2, 2);
    exp_output.setValues({{{
        {5, 6},
        {8, 9},
    }}});
    Tensor<float, 4> exp_grad_input(1, in_channels, 3, 3);
    exp_grad_input.setValues({{{
        {0, 0, 0},
        {0, 1, 1},
        {0, 1, 1}
    }}});
    auto maxpool2d = MaxPool2d<float>(2, 2); 

    // Act
    output = maxpool2d.forward(input);
    Tensor<float, 4> grad_out = output;
    grad_out.setConstant(1);
    auto grad_input = maxpool2d.backward(grad_out);

    assert_equal<float, 4>(exp_output, output);
    assert_equal<float, 4>(exp_grad_input, grad_input);
}