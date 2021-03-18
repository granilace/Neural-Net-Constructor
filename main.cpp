#include <iostream>
#include <Eigen/Dense>
#include "src/utils/Tensor.h"
#include "src/utils/Parameter.h"

using Eigen::MatrixXd;
using Eigen::Matrix;
using Eigen::Dynamic;

void test_eigen() {
    MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;
}

int main()
{
    test_eigen();
    Tensor<float> tensor(3, 4);
    Tensor<float> vec(4, 1);
    Parameter<float> parameter(tensor);
    Tensor<float> result = (tensor * vec);
}