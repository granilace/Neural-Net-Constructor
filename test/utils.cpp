#include <gtest/gtest.h>

#include "utils/DataLoader.h"
#include "utils/Dataset.h"
#include "utils/Parameter.h"
#include "utils/Tensor.h"
#include <string>

TEST(CsvDataset, ReadWithLabels) {
    CsvDataset dataset(std::string(TEST_SOURCE_DIR) + "data.csv", true, 1);
    Tensor<float, 2> trueData;
    Tensor<int, 2> trueLabels;
    trueData.resize(4, 3);
    trueLabels.resize(4, 1);
    trueData.setValues({{3.5, 4, 2.2},
                        {7.592, 8.3, 1.1},
                        {2.9, 4.4444, 2.53},
                        {1.592, 2.8, 3.9}});
    trueLabels.setValues({{1}, {0}, {1}, {0}});

    ASSERT_TRUE(dataset.size() == 4 && dataset.nFeatures() == 3);
    ASSERT_TRUE(dataset.hasLabel());


    for (int i = 0; i != dataset.size(); ++i) {
        auto element = dataset.getItem(i);
        Tensor<float, 1> trueDataRow = trueData.chip(i, 0);
        Tensor<int, 1> trueLabelsRow = trueLabels.chip(i, 0);
        ASSERT_TRUE(isApprox(trueDataRow, element.first));
        ASSERT_TRUE(isApprox(trueLabelsRow, element.second));
    }
}

TEST(CsvDataset, ReadWithoutLabels) {
    CsvDataset dataset(std::string(TEST_SOURCE_DIR) + "data.csv", false, 1);
    Tensor<float, 2> trueData;
    trueData.resize(4, 4);
    trueData.setValues({{3.5, 4, 2.2, 1},
            {7.592, 8.3, 1.1, 0},
            {2.9, 4.4444, 2.53, 1},
            {1.592, 2.8, 3.9, 0}});

    ASSERT_TRUE(dataset.size() == 4 && dataset.nFeatures() == 4);
    ASSERT_TRUE(!dataset.hasLabel());

    for (int i = 0; i != dataset.size(); ++i) {
        auto element = dataset.getItem(i);
        Tensor<float, 1> trueDataRow = trueData.chip(i, 0);
        ASSERT_TRUE(isApprox(trueDataRow, element.first));
        ASSERT_TRUE(element.second.size() == 0);
    }
}

TEST(CsvDatasetLoader, BatchWithLabels) {
    CsvDataset dataset(std::string(TEST_SOURCE_DIR) + "data.csv", true, 1);
    CsvDatasetLoader loader1(&dataset, 1);
    CsvDatasetLoader loader2(&dataset, 2);
    CsvDatasetLoader loader3(&dataset, 3);
    Tensor<float, 2> trueData;
    Tensor<int, 2> trueLabels;
    trueData.resize(4, 3);
    trueLabels.resize(4, 1);
    trueData.setValues({{3.5, 4, 2.2},
            {7.592, 8.3, 1.1},
            {2.9, 4.4444, 2.53},
            {1.592, 2.8, 3.9}});
    trueLabels.setValues({{1}, {0}, {1}, {0}});

    for (int i = 0; i != 10; ++i) {
        int idx = i % dataset.size();
        auto trueElement = dataset.getItem(idx);
        ASSERT_TRUE(loader1.nextBatchIndex() == (i % loader1.size()));
        auto batch = loader1.nextBatch();
        Tensor<float, 2> trueDataRow = trueData.slice(std::array<int, 2>({idx, 0}), std::array<int, 2>({1, dataset.size()}));
        Tensor<int, 2> trueLabelsRow = trueLabels.slice(std::array<int, 2>({idx, 0}), std::array<int, 2>({1, dataset.size()}));
        ASSERT_TRUE(isApprox(trueDataRow, batch.first));
        ASSERT_TRUE(isApprox(trueLabelsRow, batch.second));
    }

    for (int i = 0; i != 100000; ++i) {
        int idx = (i % (dataset.size() / 2)) * 2;
        auto trueElement = dataset.getItem(idx);
        ASSERT_TRUE(loader2.nextBatchIndex() == (i % loader2.size()));
        auto batch = loader2.nextBatch();
        Tensor<float, 2> trueDataRow = trueData.slice(std::array<int, 2>({idx, 0}), std::array<int, 2>({2, dataset.size()}));
        Tensor<int, 2> trueLabelsRow = trueLabels.slice(std::array<int, 2>({idx, 0}), std::array<int, 2>({2, dataset.size()}));
        ASSERT_TRUE(isApprox(trueDataRow, batch.first));
        ASSERT_TRUE(isApprox(trueLabelsRow, batch.second));
    }

    for (int i = 0; i != 100000; ++i) {
        int idx = (i % (dataset.size() / 3)) * 3;
        auto trueElement = dataset.getItem(idx);
        ASSERT_TRUE(loader3.nextBatchIndex() == (i % loader3.size()));
        auto batch = loader3.nextBatch();
        Tensor<float, 2> trueDataRow = trueData.slice(std::array<int, 2>({idx, 0}), std::array<int, 2>({3, dataset.size()}));
        Tensor<int, 2> trueLabelsRow = trueLabels.slice(std::array<int, 2>({idx, 0}), std::array<int, 2>({3, dataset.size()}));
        ASSERT_TRUE(isApprox(trueDataRow, batch.first));
        ASSERT_TRUE(isApprox(trueLabelsRow, batch.second));
    }
}

TEST(CsvDatasetLoader, BatchWithoutLabels) {
    CsvDataset dataset(std::string(TEST_SOURCE_DIR) + "data.csv", false, 1);
    CsvDatasetLoader loader1(&dataset, 1);
    CsvDatasetLoader loader2(&dataset, 2);
    CsvDatasetLoader loader3(&dataset, 3);
    Tensor<float, 2> trueData;
    trueData.resize(4, 4);
    trueData.setValues({{3.5, 4, 2.2, 1},
            {7.592, 8.3, 1.1, 0},
            {2.9, 4.4444, 2.53, 1},
            {1.592, 2.8, 3.9, 0}});

    for (int i = 0; i != 10; ++i) {
        int idx = i % dataset.size();
        auto trueElement = dataset.getItem(idx);
        ASSERT_TRUE(loader1.nextBatchIndex() == (i % loader1.size()));
        auto batch = loader1.nextBatch();
        Tensor<float, 2> trueDataRows = trueData.slice(std::array<int, 2>({idx, 0}), std::array<int, 2>({1, dataset.size()}));
        ASSERT_TRUE(isApprox(trueDataRows, batch.first));
        ASSERT_TRUE(batch.second.size() == 0);
    }

    for (int i = 0; i != 10; ++i) {
        int idx = (i % (dataset.size() / 2)) * 2;
        auto trueElement = dataset.getItem(idx);
        ASSERT_TRUE(loader2.nextBatchIndex() == (i % loader2.size()));
        auto batch = loader2.nextBatch();
        Tensor<float, 2> trueDataRows = trueData.slice(std::array<int, 2>({idx, 0}), std::array<int, 2>({2, dataset.size()}));
        ASSERT_TRUE(isApprox(trueDataRows, batch.first));
        ASSERT_TRUE(batch.second.size() == 0);
    }

    for (int i = 0; i != 10; ++i) {
        int idx = (i % (dataset.size() / 3)) * 3;
        auto trueElement = dataset.getItem(idx);
        ASSERT_TRUE(loader3.nextBatchIndex() == (i % loader3.size()));
        auto batch = loader3.nextBatch();
        Tensor<float, 2> trueDataRows = trueData.slice(std::array<int, 2>({idx, 0}), std::array<int, 2>({3, dataset.size()}));
        ASSERT_TRUE(isApprox(trueDataRows, batch.first));
        ASSERT_TRUE(batch.second.size() == 0);
    }
}

TEST(Tensor, Serialization) {
    Tensor<float, 2> tensor(10, 4);
    tensor.setConstant(1);
    tensor(0, 0) = 5.24;
    tensor(1, 2) = -1.3;
    tensor(5, 1) = 0.0;

    {
        std::ofstream out_file(std::string(TEST_SOURCE_DIR) + "dump", std::ios::binary);
        save_weights(tensor, out_file);
    }
    Tensor<float, 2> loaded_tensor(10, 4);
    std::ifstream in_file(std::string(TEST_SOURCE_DIR) + "dump", std::ios::binary);
    load_weights(loaded_tensor, in_file);

    ASSERT_NEAR(loaded_tensor(0, 0), 5.24, 1e-5);
    ASSERT_NEAR(loaded_tensor(1, 2), -1.3, 1e-5);
    ASSERT_NEAR(loaded_tensor(5, 1), 0.0, 1e-5);
    ASSERT_NEAR(loaded_tensor(8, 3), 1.0, 1e-5);
}

TEST(Parameter, Serialization) {
    Tensor<float, 2> tensor(10, 4);
    tensor.setConstant(1);
    tensor(0, 0) = 5.24;
    tensor(1, 2) = -1.3;
    tensor(5, 1) = 0.0;
    Parameter<float> parameter(tensor);
    parameter.gradient(3, 3) = -2.1;
    parameter.gradient(2, 0) = 4.12;

    {
        std::ofstream out_file(std::string(TEST_SOURCE_DIR) + "dump", std::ios::binary);
        parameter.dump(out_file);
    }
    Parameter<float> loaded_parameter(Tensor<float, 2>(10, 4));
    std::ifstream in_file(std::string(TEST_SOURCE_DIR) + "dump", std::ios::binary);
    loaded_parameter.load(in_file);

    ASSERT_NEAR(loaded_parameter.tensor(0, 0), 5.24, 1e-5);
    ASSERT_NEAR(loaded_parameter.tensor(1, 2), -1.3, 1e-5);
    ASSERT_NEAR(loaded_parameter.tensor(5, 1), 0.0, 1e-5);
    ASSERT_NEAR(loaded_parameter.tensor(8, 3), 1.0, 1e-5);

    ASSERT_NEAR(loaded_parameter.gradient(0, 0), 0.0, 1e-5);
    ASSERT_NEAR(loaded_parameter.gradient(3, 3), -2.1, 1e-5);
    ASSERT_NEAR(loaded_parameter.gradient(2, 0), 4.12, 1e-5);
}

TEST(ImageDataset, WithLabels) {
    ImageDataset imageDataset(std::string(TEST_SOURCE_DIR) + "imgs", std::string(TEST_SOURCE_DIR) + "imgs_labels.csv", ',');

    auto first_item = imageDataset.getItem(0);
    auto first_image = first_item.first;
    auto first_image_dimensions = first_image.dimensions();
    ASSERT_TRUE(first_image_dimensions.size() == 3);
    ASSERT_TRUE(first_image_dimensions[0] == 32);
    ASSERT_TRUE(first_image_dimensions[1] == 32);
    ASSERT_TRUE(first_image_dimensions[2] == 3);
    ASSERT_NEAR(first_image(0, 0, 0), 59.0, 1e-7);
    ASSERT_NEAR(first_image(0, 0, 1), 62.0, 1e-7);
    ASSERT_NEAR(first_image(0, 0, 2), 63.0, 1e-7);
    ASSERT_NEAR(first_image(11, 12, 0), 213.0, 1e-7);
    ASSERT_NEAR(first_image(11, 12, 1), 150.0, 1e-7);
    ASSERT_NEAR(first_image(11, 12, 2), 59.0, 1e-7);
    auto first_label = first_item.second;
    auto first_label_dimensions = first_label.dimensions();
    ASSERT_TRUE(first_label_dimensions.size() == 1);
    ASSERT_TRUE(first_label_dimensions[0] == 1);
    ASSERT_TRUE(first_label(0) == 0);

    auto fourth_item = imageDataset.getItem(3);
    auto fourth_image = fourth_item.first;
    auto fourth_image_dimensions = fourth_image.dimensions();
    ASSERT_NEAR(fourth_image(19, 7, 0), 111.0, 1e-7);
    ASSERT_NEAR(fourth_image(19, 7, 1), 96.0, 1e-7);
    ASSERT_NEAR(fourth_image(19, 7, 2), 72.0, 1e-7);
    auto fourth_label = fourth_item.second;
    auto fourth_label_dimensions = fourth_label.dimensions();
    ASSERT_TRUE(fourth_label_dimensions.size() == 1);
    ASSERT_TRUE(fourth_label_dimensions[0] == 1);
    ASSERT_TRUE(fourth_label(0) == 2);
}

TEST(ImageDatasetLoader, WithLabels) {
    ImageDataset imageDataset(std::string(TEST_SOURCE_DIR) + "imgs", std::string(TEST_SOURCE_DIR) + "imgs_labels.csv", ',');

    ImageDatasetLoader imageDatasetLoader(&imageDataset, 4);
    auto batch = imageDatasetLoader.nextBatch();
    auto batch_images = batch.first;
    auto batch_labels = batch.second;
    ASSERT_TRUE(batch_images.dimension(0) == 4);
    for (size_t image_idx = 0; image_idx < 4; ++image_idx) {
        auto expected_item = imageDataset.getItem(image_idx);

        auto expected_image = expected_item.first;
        Tensor<float, 3> provided_image = batch_images.chip(image_idx, 0);
        ASSERT_TRUE(isApprox(provided_image, expected_image));

        auto expected_label = expected_item.second;
        Tensor<int, 1> provided_label = batch_labels.chip(image_idx, 0);
        ASSERT_TRUE(isApprox(provided_label, expected_label));
    }

    batch = imageDatasetLoader.nextBatch();
    batch_images = batch.first;
    batch_labels = batch.second;
    std::cout << "WTF " << batch_images.dimension(0) << std::endl;
    ASSERT_TRUE(batch_images.dimension(0) == 4);
    for (size_t image_idx = 4; image_idx < 8; ++image_idx) {
        auto expected_item = imageDataset.getItem(image_idx);

        auto expected_image = expected_item.first;
        Tensor<float, 3> provided_image = batch_images.chip(image_idx - 4, 0);
        ASSERT_TRUE(isApprox(provided_image, expected_image));

        auto expected_label = expected_item.second;
        Tensor<int, 1> provided_label = batch_labels.chip(image_idx - 4, 0);
        ASSERT_TRUE(isApprox(provided_label, expected_label));
    }

    ASSERT_TRUE(imageDatasetLoader.nextBatchIndex() == 0);
}
