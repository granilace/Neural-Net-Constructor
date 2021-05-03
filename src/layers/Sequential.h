//
// Created by Sergei Troshin on 18.03.2021.
//

#ifndef NEURAL_NET_CONSTRUCTOR1_SEQUENTIAL_H
#define NEURAL_NET_CONSTRUCTOR1_SEQUENTIAL_H

#include "Identity.h"
#include "Layer.h"
#include <any>
#include <initializer_list>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

template <std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
for_each(std::tuple<Tp...>&, FuncT) {}

template <std::size_t I = 0, typename FuncT, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), void>::type for_each(std::tuple<Tp...>& t, FuncT f) {
    f(std::get<I>(t));
    for_each<I + 1, FuncT, Tp...>(t, f);
}

template <std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
for_each_reversed(std::tuple<Tp...>&, FuncT) {}

template <std::size_t I = 0, typename FuncT, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), void>::type for_each_reversed(std::tuple<Tp...>& t,
                                                   FuncT f) {
    f(std::get<sizeof...(Tp) - 1 - I>(t));
    for_each_reversed<I + 1, FuncT, Tp...>(t, f);
}

template <int InNumDims = 2, int OutNumDims = 2, class... Layers>
class Sequential
    : public Layer<typename std::tuple_element<
                       0, std::tuple<Layers...>>::type::in_element_type,
                   typename std::tuple_element<
                       sizeof...(Layers) - 1,
                       std::tuple<Layers...>>::type::out_element_type,
                   InNumDims, OutNumDims> {
    using base_class =
        Layer<typename std::tuple_element<
                  0, std::tuple<Layers...>>::type::in_element_type,
              typename std::tuple_element<
                  sizeof...(Layers) - 1,
                  std::tuple<Layers...>>::type::out_element_type,
              InNumDims, OutNumDims>;
    using typename base_class::in_element_type;
    using typename base_class::out_element_type;

  public:
    static constexpr auto layer_type = LayerType::SequentialLayer;

    Sequential(Layers&&... layers) : layers({layers...}) {}

    Tensor<out_element_type, OutNumDims>
    forward(Tensor<in_element_type, InNumDims> const& X) override {
        std::cout << "Inside Sequential::forward" << std::endl;
        any t = X;
        std::cout << "After any t = X" << std::endl;
        for_each(layers, [&t](auto& layer) {
            t = layer.forward(
                any_cast<Tensor<typename std::remove_reference<decltype(
                                    layer)>::type::in_element_type,
                                std::remove_reference<decltype(
                                    layer)>::type::in_num_dims>>(t));
        });
        return any_cast<Tensor<out_element_type, OutNumDims>>(t);
    }

    Tensor<in_element_type, InNumDims>
    backward(Tensor<out_element_type, OutNumDims> const& grad) override {
        any t = grad;
        for_each_reversed(layers, [&t](auto& layer) {
            t = layer.backward(
                any_cast<Tensor<
                    typename std::remove_reference<decltype(
                        layer)>::type::in_element_type,
                    std::remove_reference<decltype(layer)>::type::in_num_dims>>(
                    t));
        });
        return any_cast<Tensor<in_element_type, InNumDims>>(t);
    }

    void save_weights(std::ofstream& file) override {
        for_each(layers, [&file](auto& layer) { layer.save_weights(file); });
    }

    void load_weights(std::ifstream& file) override {
        for_each(layers, [&file](auto& layer) { layer.load_weights(file); });
    }

    void update(Optimizer& optimizer) override {
        for_each(layers,
                 [&optimizer](auto& layer) { layer.update(optimizer); });
    }

    void init_weights() {
        for_each(layers, [](auto& layer) { layer.init_weights(); });
    }

    void train() {
      Sequential::is_training = true;
      for_each(layers,
                 [](auto& layer) { layer.train(); });
    }
    void eval() {
      Sequential::is_training = false;
      for_each(layers,
                 [](auto& layer) { layer.eval(); });
    }

  private:
    std::tuple<Layers...> layers;
};

#endif // NEURAL_NET_CONSTRUCTOR1_SEQUENTIAL_H
