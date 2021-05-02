//
// Created by Sergei Troshin on 18.03.2021.
//

#ifndef NEURAL_NET_CONSTRUCTOR1_TENSOR_H
#define NEURAL_NET_CONSTRUCTOR1_TENSOR_H

#define SEED 1

#include <array>
#include <fstream>
#include <iostream>
#include <random>
#include <unsupported/Eigen/CXX11/Tensor>

static std::mt19937 gen(SEED);

using Eigen::Tensor;
template <typename T>
using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using ArrayType = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T, int NumDims>
Tensor<T, NumDims> transpose(const Tensor<T, NumDims>& t,
                             Eigen::array<int, 2> shuffling = {1, 0}) {
    return t.shuffle(shuffling);
}

template <typename T, int NumDims>
Tensor<T, NumDims> matmul(const Tensor<T, NumDims>& lhs,
                          const Tensor<T, NumDims>& rhs) {
    return lhs.contract(rhs, Eigen::array<Eigen::IndexPair<int>, 1>(
                                 {Eigen::IndexPair<int>(1, 0)}));
}

// template <typename T, int NumDims>
// Tensor<T, NumDims> arange(const Tensor<T, NumDims>& t) {
//     size_t Ndim = t.NumDimensions;
//     size_t NumElem = t.size();
//     const auto& d = t.dimensions();
//     for (int i = 0; i < )
// }

template <typename T, int NumDims>
void save_weights(Tensor<T, NumDims>& t, std::ofstream& file) {
    int rank = t.rank();
    file.write((char*)(&rank), sizeof(rank));
    for (size_t idx : t.dimensions()) {
        file.write((char*)(&idx), sizeof(idx));
    }
    auto data = t.data();
    file.write(
        (char*)data,
        t.size() *
            sizeof(typename std::remove_reference<decltype(t)>::type::Scalar));
}

template <typename T, int NumDims>
void load_weights(Tensor<T, NumDims>& t, std::ifstream& file) {
    int rank;
    file.read((char*)(&rank), sizeof(rank));
    assert(rank == t.rank());
    for (size_t ref_idx : t.dimensions()) {
        size_t idx;
        file.read((char*)(&idx), sizeof(idx));
        assert(idx == ref_idx);
    }

    auto data = t.data();
    file.read(
        (char*)data,
        t.size() *
            sizeof(typename std::remove_reference<decltype(t)>::type::Scalar));
}

template <typename T> void save_weights(MatrixType<T>& t, std::ofstream& file) {
    auto rows = t.rows();
    auto cols = t.cols();
    file.write((char*)(&rows), sizeof(rows));
    file.write((char*)(&cols), sizeof(cols));
    auto data = t.data();
    file.write(
        (char*)data,
        t.size() *
            sizeof(typename std::remove_reference<decltype(t)>::type::Scalar));
}

template <typename T> void load_weights(MatrixType<T>& t, std::ifstream& file) {
    typename std::remove_reference<decltype(t)>::type::Index rows = 0, cols = 0;
    file.read((char*)(&rows), sizeof(rows));
    file.read((char*)(&cols), sizeof(cols));
    t.resize(rows, cols);
    auto data = t.data();
    file.read(
        (char*)data,
        t.size() *
            sizeof(typename std::remove_reference<decltype(t)>::type::Scalar));
}

template <typename Derived>
auto get_value(const Eigen::TensorBase<Derived>& t) {
    return Tensor<float, 0>(t.eval())(0);
}

template <typename Scalar, int rank, typename sizeType>
auto Tensor_to_Matrix(const Tensor<Scalar, rank>& tensor, const sizeType rows,
                      const sizeType cols) {
    return Eigen::Map<const MatrixType<Scalar>>(tensor.data(), rows, cols);
}

template <typename Scalar>
auto Tensor_to_Matrix(const Tensor<Scalar, 2>& tensor) {
    return Eigen::Map<const MatrixType<Scalar>>(
        tensor.data(), tensor.dimension(0), tensor.dimension(1));
}

template <typename Scalar>
auto Tensor_to_Array(const Tensor<Scalar, 2>& tensor) {
    return Eigen::Map<const ArrayType<Scalar>>(
        tensor.data(), tensor.dimension(0), tensor.dimension(1));
}

template <typename Scalar, typename... Dims>
auto Matrix_to_Tensor(const MatrixType<Scalar>& matrix, Dims... dims) {
    constexpr int rank = sizeof...(Dims);
    return Eigen::TensorMap<Tensor<const Scalar, rank>>(matrix.data(),
                                                        {dims...});
}

template <typename Scalar>
auto Matrix_to_Tensor(const MatrixType<Scalar>& matrix) {
    return Eigen::TensorMap<Tensor<const Scalar, 2>>(
        matrix.data(), {matrix.rows(), matrix.cols()});
}

template <typename T, int NumDims>
bool isApprox(const Tensor<T, NumDims>& t1, const Tensor<T, NumDims>& t2) {
    MatrixType<T> mt1 = Tensor_to_Matrix(t1, 1, (int)t1.size());
    MatrixType<T> mt2 = Tensor_to_Matrix(t2, 1, (int)t2.size());
    return mt1.isApprox(mt2);
}

template <typename T, typename... Dims>
Tensor<T, sizeof...(Dims)> NormTensor(Dims... dims) {
    std::normal_distribution<T> distribution(0, 1);
    return Tensor<T, sizeof...(Dims)>(dims...).unaryExpr([&distribution](T) {
        return distribution(gen);
    });
}

template <typename T, T from, T to, typename... Dims>
Tensor<T, sizeof...(Dims)>
UniformTensor(Dims... dims) {
    std::uniform_real_distribution<T> distribution(from, to);
    return Tensor<T, sizeof...(Dims)>(dims...).unaryExpr([&distribution](T) {
        return distribution(gen);
    });
}

template <typename T, size_t Dims>
Tensor<T, Dims> 
tensor_fill_bernoulli(Tensor<T, Dims> t, float p) {
    std::bernoulli_distribution distribution(p);
    return t.unaryExpr([&distribution](T) {
        return static_cast<T>(distribution(gen));
    });
}

#endif // NEURAL_NET_CONSTRUCTOR1_TENSOR_H
