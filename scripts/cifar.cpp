#include <iostream>
#include <experimental/filesystem>
#include <string>
#include "layers/Activations.h"
#include "layers/Flatten.h"
#include "layers/Linear.h"
#include "layers/Conv2d.h"
#include "layers/Sequential.h"
#include "layers/MaxPool2d.h"
#include "utils/DataLoader.h"
#include "utils/Dataset.h"
#include <sys/stat.h>
#include <dirent.h>


struct RunOptions {
    float learning_rate = 0.001;
    size_t epochs = 5;
    size_t batch_size = 8;
};
struct DatasetOptions {
    std::string imgs_dir_path;
    std::string labels_csv_path;
};

struct RunOptions ParseArgs(int argc, char** argv) {
    struct RunOptions run_options;
    int arg_idx = 1;
    while (arg_idx < argc) {
        auto argument = std::string(argv[arg_idx]);
        if (argument == "--learning-rate") {
            run_options.learning_rate = std::stof(argv[arg_idx + 1]);
            ++arg_idx;
        } else if (argument == "--epochs") {
            run_options.epochs = std::stoul(argv[arg_idx]);
            ++arg_idx;
        } else if (argument == "--batch-size") {
            run_options.batch_size = std::stoul(argv[arg_idx]);
            ++arg_idx;
        } else {
            std::cout << "Unrecognized argument: " << argument << std::endl;
        }
        ++arg_idx;
    }

    return run_options;
}


int main(int argc, char** argv) {
    auto run_options = ParseArgs(argc, argv);

    auto test_dataset = ImageDataset("data/cifar-10/prepared/test", "data/cifar-10/prepared/labels.csv", ',');
    auto test_dataset_loader = ImageDatasetLoader(&test_dataset, run_options.batch_size);
    auto train_dataset = ImageDataset("data/cifar-10/prepared/test", "data/cifar-10/prepared/labels.csv", ',');
    auto train_dataset_loader = ImageDatasetLoader(&train_dataset, run_options.batch_size);

    auto model = Sequential<4, 2, Conv2d<float>, MaxPool2d<float>, Conv2d<float>, MaxPool2d<float>, Conv2d<float>, Flatten<float>, Linear<float>, Softmax<float>>(
        Conv2d<float>(3, 4, 5, 5),
        MaxPool2d<float>(2, 2),
        Conv2d<float>(4, 8, 5, 5),
        MaxPool2d<float>(2, 2),
        Conv2d<float>(8, 16, 5, 5),
        Flatten<float>(),
        Linear<float>(16, 10),
        Softmax<float>()
    );
    auto opt = Optimizer(run_options.learning_rate);

    size_t epoch = 0;
    while (epoch < run_options.epochs) {
        auto batch = train_dataset_loader.nextBatch();
        Tensor<float, 4> batch_imgs = batch.first;
        // std::cout << "batch imgs prepared " << batch_imgs.dimension(0) << ", " << batch_imgs.dimension(1) << ", " << batch_imgs.dimension(2) << ", " << batch_imgs.dimension(3) << std::endl;
        Tensor<float, 2> forward_result = model.forward(batch_imgs);
        // ::cout << "after model.forward(batch_imgs)" << std::endl;
        // std::cout << "forward_result: " << forward_result.dimension(0) << ", " << forward_result.dimension(1) << std::endl;
        // model.backward(grad);
        // model.update(opt);
        if (train_dataset_loader.nextBatchIndex() == 0) {
            std::cout << "Epoch " << epoch << " finished" << std::endl;
            ++epoch;
        }
    }

    std::cout << "Evaluation succeeded" << std::endl;
    return 0;
}
