//
// Created by Sergei Troshin on 18.03.2021.
//

#ifndef NEURAL_NET_CONSTRUCTOR1_TENSOR_H
#define NEURAL_NET_CONSTRUCTOR1_TENSOR_H

#define SEED 1

#include <fstream>
#include <iostream>
#include <random>
#include <Eigen/Dense>

static std::mt19937 gen(SEED);

using Eigen::Dynamic;

template<typename T>
class Tensor : public Eigen::Matrix<T, Dynamic, Dynamic> {
 public:
    Tensor() { }
    Tensor(size_t h, size_t w) : Eigen::Matrix<T, Dynamic, Dynamic>(h, w) { }
    template<typename Derived>
    Tensor(const Eigen::MatrixBase<Derived> & m) : Eigen::Matrix<T, Dynamic, Dynamic>(m) { }
    template<typename Derived>
    Tensor(Eigen::MatrixBase<Derived> && m) : Eigen::Matrix<T, Dynamic, Dynamic>(std::move(m)) { }

    void dump(std::ofstream & file) {
        typename Tensor::Index rows = this->rows(), cols = this->cols();
        file.write((char*) (&rows), sizeof(rows));
        file.write((char*) (&cols), sizeof(cols));

        auto data = this->data();
        file.write((char*) data, rows * cols * sizeof(typename Tensor::Scalar));
    }

    void load(std::ifstream & file) {
        typename Tensor::Index rows = 0, cols = 0;
        file.read((char*) (&rows), sizeof(rows));
        file.read((char*) (&cols), sizeof(cols));
        this->resize(rows, cols);

        auto data = this->data();
        file.read((char *) data, rows * cols * sizeof(typename Tensor::Scalar));
    }
};

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
