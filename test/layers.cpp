#include <gtest/gtest.h>

#include "utils/Tensor.h"
#include "utils/Parameter.h"
#include "layers/Identity.h"
#include "layers/Linear.h"
#include "layers/Conv2d.h"
#include "layers/Sequential.h"
#include "layers/MaxPool2d.h"

static const float kEps = 1e-7;

TEST(Sequential, Identity) {
    Tensor<float, 2> tensor(3, 4);
    tensor.setConstant(1);

    Tensor<float, 2> vec(4, 1);
    vec.setConstant(1);

    Parameter<float> parameter(tensor);
    Tensor<float, 2> input = matmul(tensor, vec);

    auto model = Sequential(
        Identity<float>(),
        Identity<float>()
    );
    Tensor<float, 2> loss = model.forward(input);
    model.backward(loss);
    ASSERT_NEAR(get_value(loss.sum().eval()), 12.0, kEps);
}

TEST(Tmp, tmp) {
    Eigen::Matrix<float, 10, 4> input;
    input.setConstant(1);

    Eigen::Matrix<float, 4, 3> weight;
    weight.setConstant(1);

    Eigen::Matrix<float, 3, 1> bias;
    bias.setConstant(1);

    Eigen::VectorXf vector_bias(3);

    auto mul_result = input * weight;
    // std::cout << mul_result.rows() << " " << mul_result.cols() << std::endl;
    auto bias_sum_result = bias.rowwise().sum();
    // std::cout << bias_sum_result.rows() << " " << bias_sum_result.cols() << std::endl;
    auto tmp = mul_result.rowwise() + bias.transpose();
    // std::cout << tmp.rows() << " " << tmp.cols() << std::endl;
    // std::cout << tmp << std::endl;
}

TEST(Linear, Forward) {
    Linear<float> linear(4, 3);
    Tensor<float, 2> input(10, 4);
    input.setConstant(1);
    auto output = linear.forward(input);
    auto output_row_sum = output.sum(std::array<int, 1>{1});
    ASSERT_EQ(output.dimension(0), 10);
    ASSERT_EQ(output.dimension(1), 3);
    ASSERT_NEAR(get_value(output_row_sum.minimum().eval()), get_value(output_row_sum.maximum().eval()), kEps); // check that all output rows are the same
}

TEST(Linear, Serialization) {
    Linear<float> linear(4, 3);
    linear.init_weights();
    // std::cout << "linear:" << std::endl;
    // std::cout << "weights_: " << linear.weights_.tensor << std::endl;
    // std::cout << "bias_: " << linear.bias_.tensor << std::endl;

    {
        std::ofstream out_file(std::string(TEST_SOURCE_DIR) + "dump", std::ios::binary);
        linear.save_weights(out_file);
    }
    Linear<float> loaded_linear(4, 3);
    std::ifstream in_file(std::string(TEST_SOURCE_DIR) + "dump", std::ios::binary);
    loaded_linear.load_weights(in_file);
    // std::cout << "loaded_linear:" << std::endl;
    // std::cout << "weights_: " << loaded_linear.weights_.tensor << std::endl;
    // std::cout << "bias_: " << loaded_linear.bias_.tensor << std::endl;

    Tensor<float, 2> input(2, 4);
    input.setConstant(1);

    auto linear_output = linear.forward(input);
    // std::cout << "linear_output:" << linear_output << std::endl;
    auto loaded_linear_output = loaded_linear.forward(input);
    auto diff = linear_output - loaded_linear_output;
    Tensor<float, 2> diff_squared = diff.unaryExpr([](float d) {return d * d;}).eval();
    float diff_squared_sum = get_value(diff_squared.sum().eval());

    ASSERT_NEAR(diff_squared_sum, 0.0, kEps);
}

TEST(Sequential, Serialization) {
    auto model = Sequential(
        Linear<float>(4, 3),
        Identity<float>(),
        Linear<float>(3, 2)
    );

    {
        std::ofstream out_file(std::string(TEST_SOURCE_DIR) + "dump", std::ios::binary);
        model.save_weights(out_file);
    }
    auto loaded_model = Sequential(
        Linear<float>(4, 3),
        Identity<float>(),
        Linear<float>(3, 2)
    );
    std::ifstream in_file(std::string(TEST_SOURCE_DIR) + "dump", std::ios::binary);
    loaded_model.load_weights(in_file);
    // std::cout << "loaded_linear:" << std::endl;
    // std::cout << "weights_: " << loaded_linear.weights_.tensor << std::endl;
    // std::cout << "bias_: " << loaded_linear.bias_.tensor << std::endl;

    Tensor<float, 2> input(2, 4);
    input.setConstant(1);

    auto model_output = model.forward(input);
    auto loaded_model_output = loaded_model.forward(input);
    auto diff = model_output - loaded_model_output;
    auto diff_squared = diff.unaryExpr([](float d) {return d * d;});
    auto diff_squared_sum = get_value(diff.sum());

    ASSERT_NEAR(diff_squared_sum, 0.0, kEps);
}

TEST(Conv2dMaxPooltest, Conv2dMaxpool2d) {
    // size_t in_channels, size_t out_channels, size_t kernel_height, size_t kernel_width

    size_t in_channels = 1;
    size_t out_channels = 3;
    size_t kernel_height = 3;
    size_t kernel_width = 3;

    auto conv = Conv2d<float>(in_channels, out_channels, kernel_height, kernel_width); 
    Tensor<float, 4> input(1, in_channels, 3, 3); // batch size x h x w x in_channels
    Tensor<float, 4> kernel(in_channels, out_channels, kernel_height, kernel_width);  
    Tensor<float, 1> bias(out_channels);  

    Tensor<float, 4> output = conv2d<float>(input, kernel, bias);
    std::cout << "output" << output;

    output = conv.forward(input);

    MaxPool2d<float> mp = MaxPool2d<float>(2, 2);
    output = mp.forward(input);
}