//
// Created by Sergei Troshin on 18.03.2021.
//

#ifndef NEURAL_NET_CONSTRUCTOR1_TENSOR_H
#define NEURAL_NET_CONSTRUCTOR1_TENSOR_H

#include <iostream>
#include <Eigen/Dense>
using Eigen::Matrix;
using Eigen::Dynamic;

template<typename T>
using Tensor = Eigen::Matrix<T, Dynamic, Dynamic>;

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


#endif //NEURAL_NET_CONSTRUCTOR1_TENSOR_H
