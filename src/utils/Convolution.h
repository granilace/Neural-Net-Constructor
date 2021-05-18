#pragma once
#include "Tensor.h"
#include <thread>

const int N_THREADS = 4;

template <typename T>
Tensor<T, 4> conv2d(Tensor<T, 4> const& input, Tensor<T, 4> const& weight, Tensor<T, 1> const& bias) {
    size_t bs = input.dimension(0);
    size_t h = input.dimension(2);
    size_t w = input.dimension(3);

    size_t in_channels = weight.dimension(0);
    size_t out_channels = weight.dimension(1);
    size_t kernel_height = weight.dimension(2);
    size_t kernel_width = weight.dimension(3);
    
    assert(weight.dimension(0) == input.dimension(1));
    assert(bias.dimension(0) == weight.dimension(1));

    assert(h >= kernel_height - 1);
    assert(w >= kernel_width - 1);
    size_t h_new = h - kernel_height + 1;
    size_t w_new = w - kernel_width + 1;

    Tensor<T, 4> output(bs, out_channels, h_new, w_new);
    output.setZero();

    // std::cout << "input: " << input << std::endl;
    // std::cout << "weight: " << weight << std::endl;
    int n_threads = N_THREADS;
    if (bs < n_threads)
        n_threads = bs;

    std::vector<std::thread> threads;
    for (int tid = 0; tid < n_threads; tid++) {
        threads.emplace_back([&, tid] { 
            for (size_t b = tid; b < bs; b += n_threads) {
                for (size_t ch_out = 0; ch_out < out_channels; ch_out++) {
                    for (size_t i = 0; i < h_new; i++) {
                        for (size_t j = 0; j < w_new; j++) {
                            for (size_t ch_in = 0; ch_in < in_channels; ch_in++) {
                                
                                for (size_t d_i = 0; d_i < kernel_height; d_i++) {
                                    for (size_t d_j = 0; d_j < kernel_width; d_j++) {
                                        // std::cout << "ADD " << input(b, ch_in, i + d_i, j + d_j) << " " <<  weight(ch_in, ch_out, d_i, d_j) << std::endl;
                                        // std::cout << "(b, i, j, ch_in, ch_out, d_i, d_j) = " << "(" << b << "," << i << "," << j << "," << ch_in << "," << ch_out << "," << d_i << "," << d_j << ")" << std::endl;


                                        output(b, ch_out, i, j) += input(b, ch_in, i + d_i, j + d_j)  \
                                            * weight(ch_in, ch_out, d_i, d_j);
                                    }
                                }
                                output(b, ch_out, i, j) += bias(ch_out);
                            }
                        }
                    }
                }
            }
        });

    }
    for (auto & thread : threads) {
        thread.join();
    }
    return output;
}

template <typename T>
Tensor<T, 4> grad_weight(Tensor<T, 4> const& input, Tensor<T, 4> const& grad_out) {
    // out: bs x out_channels x h_new x w_new
    size_t bs =          input.dimension(0);
    size_t in_channels = input.dimension(1);
    size_t h =           input.dimension(2);
    size_t w =           input.dimension(3);

    
    size_t out_channels = grad_out.dimension(1);
    size_t h_new =        grad_out.dimension(2);
    size_t w_new =        grad_out.dimension(3);
    size_t kernel_height =  h - h_new + 1;
    size_t kernel_width =   w - w_new + 1;

    assert(h >= kernel_height - 1);
    assert(w >= kernel_width - 1);

    Tensor<T, 4> grad_weight(in_channels, out_channels, kernel_height, kernel_width);
    grad_weight.setZero();

    int n_threads = N_THREADS;
    if (bs < n_threads)
        n_threads = bs;
    std::vector<std::thread> threads;
    for (int tid = 0; tid < n_threads; tid++) {
        threads.emplace_back([&, tid] { 
            for (size_t b = tid; b < bs; b += n_threads) {
                for (size_t i = 0; i < kernel_height; i++) {
                    for (size_t j = 0; j < kernel_width; j++) {
                        for (size_t ch_in = 0; ch_in < in_channels; ch_in++) {
                            for (size_t ch_out = 0; ch_out < out_channels; ch_out++) {
                                for (size_t d_i = 0; d_i < h_new; d_i++) {
                                    for (size_t d_j = 0; d_j < w_new; d_j++) {
                                        grad_weight(ch_in, ch_out, i, j) += input(b, ch_in, i + d_i, j + d_j) * \
                                        grad_out(b, ch_out, d_i, d_j);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
     }
     for (auto & thread : threads) {
        thread.join();
    }
    return grad_weight;
}

template <typename T>
Tensor<T, 1> grad_bias(Tensor<T, 4> const& grad_out) {
    size_t bs =           grad_out.dimension(0);
    size_t out_channels = grad_out.dimension(1);
    size_t h_new =        grad_out.dimension(2);
    size_t w_new =        grad_out.dimension(3);

    Tensor<T, 1> grad_bias(out_channels);
    grad_bias.setZero();

    int n_threads = N_THREADS;
    if (bs < n_threads)
        n_threads = bs;
    std::vector<std::thread> threads;
    for (int tid = 0; tid < n_threads; tid++) {
        threads.emplace_back([&, tid] { 
            for (size_t b = tid; b < bs; b += n_threads) {
                for (size_t i = 0; i < h_new; i++) {
                    for (size_t j = 0; j < w_new; j++) {
                        for (size_t ch_out = 0; ch_out < out_channels; ch_out++) {
                            grad_bias(ch_out) += grad_out(b, ch_out, i, j);
                        }
                    }
                }
            }
        });
    }
    for (auto & thread : threads) {
        thread.join();
    }
    return grad_bias;
}

template <typename T>
Tensor<T, 4> grad_input(Tensor<T, 4> const& weight, Tensor<T, 4> const& grad_out) {
    // out: bs x out_channels x h_new x w_new
    // weight: in_channels, out_channels, kernel_height, kernel_width
    
    size_t in_channels = weight.dimension(0);
    size_t out_channels =  weight.dimension(1);
    size_t kernel_height =  weight.dimension(2);
    size_t kernel_width =  weight.dimension(3);

    size_t bs =          grad_out.dimension(0);
    assert(int(out_channels) == grad_out.dimension(1));
    size_t h_new =           grad_out.dimension(2);
    size_t w_new =           grad_out.dimension(3);

    size_t h =           grad_out.dimension(2) + kernel_height - 1;
    size_t w =           grad_out.dimension(3) + kernel_width - 1;
    
    assert(h >= kernel_height - 1);
    assert(w >= kernel_width - 1);

    Tensor<T, 4> grad_input(bs, in_channels, h, w);
    grad_input.setZero();

    // grad_input = full_convolution(weight.T, grad_out)
    // auto weight_T_ = transpose(weight, {0, 1});
    Eigen::array<bool, 4> rev_order({false, false, true, true});
    Tensor<T, 4> weight180 = weight.reverse(rev_order);

    int n_threads = N_THREADS;
    if (bs < n_threads)
        n_threads = bs;
    std::vector<std::thread> threads;
    for (int tid = 0; tid < n_threads; tid++) {
        threads.emplace_back([&, tid] { 
            for (size_t b = tid; b < bs; b += n_threads) {
                for (size_t i = 0; i < h; i++) {
                    for (size_t j = 0; j < w; j++) {
                        for (size_t ch_in = 0; ch_in < in_channels; ch_in++) {
                            for (size_t ch_out = 0; ch_out < out_channels; ch_out++) {
                                for (size_t d_i = 0; d_i < kernel_height; d_i++) {
                                    for (size_t d_j = 0; d_j < kernel_width; d_j++) {
                                        // std::cout << "ADD " << input(b, ch_in, i + d_i, j + d_j) << " " <<  weight(ch_in, ch_out, d_i, d_j) << std::endl;
                                        size_t i_grad = i + d_i - kernel_height + 1;
                                        size_t j_grad = j + d_j - kernel_width + 1;
                                        if ((i_grad >= 0 && i_grad < h_new) && (j_grad >= 0 && j_grad < w_new)) {
                                            grad_input(b, ch_in, i, j) += grad_out(b, ch_out, i_grad, j_grad)  \
                                                * weight180(ch_in, ch_out, d_i, d_j);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
    }
    for (auto & thread : threads) {
        thread.join();
    }
    return grad_input;
}
