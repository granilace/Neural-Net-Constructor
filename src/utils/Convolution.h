#pragma once
#include "Tensor.h"

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
    size_t w_new = h - kernel_width + 1;

    Tensor<T, 4> output(bs, out_channels, h_new, w_new);
    output.setZero();

    for (size_t b = 0; b < bs; b++) {
        for (size_t i = 0; i < h_new; i++) {
            for (size_t j = 0; j < w_new; j++) {
                for (size_t ch_in = 0; ch_in < in_channels; ch_in++) {
                    for (size_t ch_out = 0; ch_out < out_channels; ch_out++) {
                        for (size_t d_i = 0; d_i < kernel_width; d_i++) {
                            for (size_t d_j = 0; d_j < kernel_height; d_j++) {
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

    for (size_t b = 0; b < bs; b++) {
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

    for (size_t b = 0; b < bs; b++) {
         for (size_t i = 0; i < h_new; i++) {
            for (size_t j = 0; j < w_new; j++) {
                for (size_t ch_out = 0; ch_out < out_channels; ch_out++) {
                    grad_bias(ch_out) += grad_out(b, ch_out, i, j);
                }
            }
        }
    }
    return grad_bias;
}
