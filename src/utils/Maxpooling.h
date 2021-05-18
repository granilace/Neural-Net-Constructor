#pragma once
#include "Tensor.h"
#include "Parallel.h"

template <typename T>
std::pair<Tensor<T, 4>, Tensor<size_t, 5>> maxpool2d(Tensor<T, 4> const& input, size_t kernel_height, size_t kernel_width) {
    size_t bs = input.dimension(0);
    size_t in_channels = input.dimension(1);
    size_t h = input.dimension(2);
    size_t w = input.dimension(3);

    size_t h_new = (h - 1) / 2 + 1;
    size_t w_new = (w - 1) / 2 + 1;

    Tensor<T, 4> output(bs, in_channels, h_new, w_new);
    Tensor<size_t, 5> max_indices(bs, in_channels, h_new, w_new, 2);
    output.setZero();

    parallelize(bs, [&](int tid, int n_threads) { 
        for (size_t b = tid; b < bs; b += n_threads) {
            for (size_t ch_in = 0; ch_in < in_channels; ch_in++) {
                for (size_t i = 0, ii = 0; i < h; i += kernel_height, ii += 1) {
                    for (size_t j = 0, jj = 0; j < w; j += kernel_width, jj += 1) {
                        T maximum = input(b, ch_in, i, j);
                        size_t mx_i = i;
                        size_t mx_j = j;
                        for (size_t d_i = 0; d_i < kernel_height; d_i += 1) {
                            if (i + d_i >= h) break;
                            for (size_t d_j = 0; d_j < kernel_width; d_j += 1) {
                                if (j + d_j >= w) break;
                                    if (maximum < input(b, ch_in, i + d_i, j + d_j)) {
                                        maximum = input(b, ch_in, i + d_i, j + d_j);
                                        mx_i = i + d_i;
                                        mx_j = j + d_j;
                                    }
                            }
                        }
                        output(b, ch_in, ii, jj) = maximum;
                        max_indices(b, ch_in, ii, jj, 0) = mx_i;
                        max_indices(b, ch_in, ii, jj, 1) = mx_j;
                    }
                }
            }
        }
    });
    return {output, max_indices};
}

template <typename T>
Tensor<T, 4> grad_maxpool(Tensor<T, 4> const& input, Tensor<T, 4> const& grad_out, Tensor<size_t, 5> const& indices, size_t kernel_height, size_t kernel_width) {
    // out: bs x out_channels x h_new x w_new
    assert(input.dimension(0) == grad_out.dimension(0));
    assert(input.dimension(1) == grad_out.dimension(1));
    assert((input.dimension(2) - 1) / 2 + 1 == grad_out.dimension(2));
    assert((input.dimension(3) - 1) / 2 + 1 == grad_out.dimension(3));
    
    size_t bs =          input.dimension(0);
    size_t in_channels = input.dimension(1);
    size_t h =           input.dimension(2);
    size_t w =           input.dimension(3);

    size_t h_new =        grad_out.dimension(2);
    size_t w_new =        grad_out.dimension(3);

    Tensor<T, 4> grad_input(bs, in_channels, h, w);
    grad_input.setZero();

    parallelize(bs, [&](int tid, int n_threads) { 
        for (size_t b = tid; b < bs; b += n_threads) {
            for (size_t ch_in = 0; ch_in < in_channels; ch_in++) {
                for (size_t i = 0; i < h_new; i += 1) {
                    for (size_t j = 0; j < w_new; j += 1) {
                        size_t i_ind = indices(b, ch_in, i, j, 0);
                        size_t j_ind = indices(b, ch_in, i, j, 1);
                        grad_input(b, ch_in, i_ind, j_ind) = grad_out(b, ch_in, i, j);
                    }
                }
            }
        }
    });
    return grad_input;
}

