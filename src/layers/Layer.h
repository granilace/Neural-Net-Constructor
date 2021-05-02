//
// Created by Sergei Troshin on 18.03.2021.
//

#ifndef NEURAL_NET_CONSTRUCTOR1_LAYER_H
#define NEURAL_NET_CONSTRUCTOR1_LAYER_H

#include "../optimizers/Optimizer.h"
#include "../utils/Parameter.h"
#include <fstream>
#include <vector>

using namespace std;

template <typename InType, typename OutType, int InNumDims, int OutNumDims>
class Layer {
  public:
    virtual ~Layer() = default;

    virtual Tensor<OutType, OutNumDims>
    forward(Tensor<InType, InNumDims> const& X) = 0;

    virtual Tensor<InType, InNumDims>
    backward(Tensor<OutType, OutNumDims> const& grad) = 0;

    virtual void update(Optimizer& optimizer) = 0;

    virtual void save_weights(std::ofstream& file) = 0;

    virtual void load_weights(std::ifstream& file) = 0;

    virtual void init_weights() = 0;

    void train() {
      is_training = true;
    }
    void eval() {
      is_training = false;
    }

    // void get_weights();

    using in_element_type = InType;
    using out_element_type = OutType;
    enum {
        in_num_dims = InNumDims,
        out_num_dims = OutNumDims,
    };
    protected:
      bool is_training = true;
};

enum LayerType { IdentityLayer = 1, LinearLayer = 2, SequentialLayer = 3,
        Conv2dLayer = 4, MaxPoolLayer = 5, ActivationLayer = 6, DropoutLayer = 7 };

#endif // NEURAL_NET_CONSTRUCTOR1_LAYER_H
