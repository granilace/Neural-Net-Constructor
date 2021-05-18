#include <nlohmann/json.hpp>
#include <fstream>
#include <iomanip>
#include <chrono>


#include "layers/Conv2d.h"
#include "layers/MaxPool2d.h"
#include "utils/Convolution.h"

// for convenience
using json = nlohmann::json;

double forward(size_t bs=1, size_t in_channels=64, size_t out_channels=64, size_t iH=128, size_t iW=128, size_t kH=3, size_t kW=3) {
    size_t kernel_height = kH;
    size_t kernel_width = kW;
    size_t H = iH;
    size_t W = iW;

    
    Tensor<float, 4> input(bs, in_channels, H, W);
    Tensor<float, 4> kernel(in_channels, out_channels, kernel_height, kernel_width);
    Tensor<float, 1> bias(out_channels);
    Tensor<float, 4> output;
    output.setZero();

    auto conv = Conv2d<float>(kernel, bias); 

    // Act
    auto start = std::chrono::steady_clock::now();
    output = conv.forward(input);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    return elapsed_seconds.count();
}

json benchmark_forward(json benchmark, json default_params_) {
    json results;
    for (auto & [key, values] : benchmark.items()) {
        json prms = default_params_;
        results[key] = {};
        for (auto & value : values) {
            std:: cout << key << " " << value << " " << std::endl;
            prms[key] = value;
            double elapced_time = forward(
                prms["bs"],
                prms["in_channels"],
                prms["out_channels"],
                prms["iH"],
                prms["iW"],
                prms["kH"],
                prms["kW"]);
            results[key].push_back({value, elapced_time});
        }
    }
    return results;
}

double bacward(size_t bs=1, size_t in_channels=64, size_t out_channels=64, size_t iH=128, size_t iW=128, size_t kH=3, size_t kW=3) {
    size_t kernel_height = kH;
    size_t kernel_width = kW;
    size_t H = iH;
    size_t W = iW;

    
    Tensor<float, 4> input(bs, in_channels, H, W);
    Tensor<float, 4> kernel(in_channels, out_channels, kernel_height, kernel_width);
    Tensor<float, 1> bias(out_channels);
    Tensor<float, 4> output;
    output.setZero();

    auto conv = Conv2d<float>(kernel, bias); 

    // Act
    output = conv.forward(input);
    auto start = std::chrono::steady_clock::now();
    conv.backward(output);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    return elapsed_seconds.count();
}

json benchmark_backward(json benchmark, json default_params_) {
    json results;
    for (auto & [key, values] : benchmark.items()) {
        json prms = default_params_;
        results[key] = {};
        for (auto & value : values) {
            std:: cout << key << " " << value << " " << std::endl;
            prms[key] = value;
            double elapced_time = forward(
                prms["bs"],
                prms["in_channels"],
                prms["out_channels"],
                prms["iH"],
                prms["iW"],
                prms["kH"],
                prms["kW"]);
            results[key].push_back({value, elapced_time});
        }
    }
    return results;
}

void save_json(json obj, std::string filename) {
    std::ofstream o(filename);
    o << std::setw(4) << obj << std::endl;
}

int main() {
    json benchmark, default_params;
    default_params = {
        {"bs",  1},
        {"in_channels", 64},
        {"out_channels", 64},
        {"iH", 128},
        {"iW", 128},
        {"kH", 3},
        {"kW", 3}
    };
    benchmark = {
            {"bs",  {1, 4, 10, 30}},
            {"in_channels", {1, 3, 10, 64}},
            {"out_channels", {1, 3, 10, 64}},
            {"iH", {3, 32, 64, 128}},
            {"kH", {1, 3, 7}}
    };
    // j["object"] = { {"currency", "USD"}, {"value", 42.99} };
    // std::ofstream o("pretty.json");
    // o << std::setw(4) << j << std::endl;

    save_json(benchmark_forward(benchmark, default_params), "conv2d.forward.json");
    save_json(benchmark_backward(benchmark, default_params), "conv2d.backward.json");

    // std::cout << forward() << std::endl;
}