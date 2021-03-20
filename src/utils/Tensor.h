//
// Created by Sergei Troshin on 18.03.2021.
//

#ifndef NEURAL_NET_CONSTRUCTOR1_TENSOR_H
#define NEURAL_NET_CONSTRUCTOR1_TENSOR_H

#define SEED 1

#include <iostream>
#include <random>
#include <Eigen/Dense>

static std::mt19937 gen(SEED);

template<typename T>
using Tensor = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

// Or
//template<typename T>
//class Tensor : public Eigen::Matrix<T, Dynamic, Dynamic> {
//public:
//    Tensor(size_t h, size_t w) : Eigen::Matrix<T, Dynamic, Dynamic>(h, w) {
//        std::cout << *this << std::endl;
//    }
//    Tensor(const Eigen::Matrix<T, Dynamic, Dynamic> & input) : Eigen::Matrix<T, Dynamic, Dynamic>(input) {
//    }
//};

template<typename T>
Tensor<T> NormTensor(size_t rows, size_t cols) {
    std::normal_distribution<T> distribution(0,1);
    return Tensor<T>(rows, cols).unaryExpr([&distribution](T) {return distribution(gen);});
}

#endif //NEURAL_NET_CONSTRUCTOR1_TENSOR_H
