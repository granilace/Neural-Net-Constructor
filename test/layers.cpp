#include <gtest/gtest.h>

#include "utils/Tensor.h"
#include "utils/Parameter.h"
#include "layers/Identity.h"
#include "layers/Linear.h"
#include "layers/Sequential.h"

static const float kEps = 1e-7;

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
    ASSERT_NEAR(loss.sum(), 12.0, kEps);
}

TEST(Tmp, tmp) {
    Eigen::Matrix<float, 10, 4> input;
    input.setOnes();

    Eigen::Matrix<float, 4, 3> weight;
    weight.setOnes();

    Eigen::Matrix<float, 3, 1> bias;
    bias.setOnes();

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
    Tensor<float> input(10, 4);
    input.setOnes();
    auto output = linear.forward(input);
    auto output_row_sum = output.rowwise().sum();
    ASSERT_EQ(output.rows(), 10);
    ASSERT_EQ(output.cols(), 3);
    ASSERT_NEAR(output_row_sum.minCoeff(), output_row_sum.maxCoeff(), kEps); // check that all output rows are the same
}

TEST(Linear, Serialization) {
    Linear<float> linear(4, 3);
    linear.init_weights();
    // std::cout << "linear:" << std::endl;
    // std::cout << "weights_: " << linear.weights_.tensor << std::endl;
    // std::cout << "bias_: " << linear.bias_.tensor << std::endl;

    {
        std::ofstream out_file(std::string(TEST_SOURCE_DIR) + "dump", std::ios::binary);
        linear.dump(out_file);
    }
    Linear<float> loaded_linear;
    std::ifstream in_file(std::string(TEST_SOURCE_DIR) + "dump", std::ios::binary);
    loaded_linear.load(in_file);
    // std::cout << "loaded_linear:" << std::endl;
    // std::cout << "weights_: " << loaded_linear.weights_.tensor << std::endl;
    // std::cout << "bias_: " << loaded_linear.bias_.tensor << std::endl;

    Tensor<float> input(2, 4);
    input.setOnes();

    auto linear_output = linear.forward(input);
    // std::cout << "linear_output:" << linear_output << std::endl;
    auto loaded_linear_output = loaded_linear.forward(input);
    auto diff = linear_output - loaded_linear_output;
    auto diff_squared = diff.unaryExpr([](float d) {return d * d;});
    auto diff_squared_sum = diff.sum();

    ASSERT_NEAR(diff_squared_sum, 0.0, kEps);
}

TEST(Sequential, Serialization) {
    auto model = Sequential<float>({
        new Linear<float>(4, 3),
        new Identity<float>(),
        new Linear<float>(3, 2),
    });

    {
        std::ofstream out_file(std::string(TEST_SOURCE_DIR) + "dump", std::ios::binary);
        model.dump(out_file);
    }
    Sequential<float> loaded_model;
    std::ifstream in_file(std::string(TEST_SOURCE_DIR) + "dump", std::ios::binary);
    loaded_model.load(in_file);
    // std::cout << "loaded_linear:" << std::endl;
    // std::cout << "weights_: " << loaded_linear.weights_.tensor << std::endl;
    // std::cout << "bias_: " << loaded_linear.bias_.tensor << std::endl;

    Tensor<float> input(2, 4);
    input.setOnes();

    auto model_output = model.forward(input);
    // std::cout << "linear_output:" << linear_output << std::endl;
    auto loaded_model_output = loaded_model.forward(input);
    auto diff = model_output - loaded_model_output;
    auto diff_squared = diff.unaryExpr([](float d) {return d * d;});
    auto diff_squared_sum = diff.sum();

    ASSERT_NEAR(diff_squared_sum, 0.0, kEps);
}
