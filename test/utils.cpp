#include <gtest/gtest.h>

#include "utils/DataLoader.h"
#include "utils/Dataset.h"
#include "utils/Tensor.h"
#include <string>

TEST(CsvDataset, ReadWithLabels) {
    CsvDataset dataset(std::string(TEST_SOURCE_DIR) + "data.csv", true, 1);
    Tensor<float> trueData;
    Tensor<int> trueLabels;
    trueData.resize(4, 3);
    trueLabels.resize(4, 1);
    trueData << 3.5, 4, 2.2,
                7.592, 8.3, 1.1,
                2.9, 4.4444, 2.53,
                1.592, 2.8, 3.9;
    trueLabels << 1, 0, 1, 0;

    ASSERT_TRUE(dataset.size() == 4 && dataset.nFeatures() == 3);
    ASSERT_TRUE(dataset.hasLabel());

    for (int i = 0; i != dataset.size(); ++i) {
        auto element = dataset.getItem(i);
        ASSERT_TRUE(trueData.block(i, 0, 1, trueData.cols()).isApprox(element.first));
        ASSERT_TRUE(trueLabels.block(i, 0, 1, trueLabels.cols()).isApprox(element.second));
    }
}

TEST(CsvDataset, ReadWithoutLabels) {
    CsvDataset dataset(std::string(TEST_SOURCE_DIR) + "data.csv", false, 1);
    Tensor<float> trueData;
    trueData.resize(4, 4);
    trueData << 3.5, 4, 2.2, 1,
            7.592, 8.3, 1.1, 0,
            2.9, 4.4444, 2.53, 1,
            1.592, 2.8, 3.9, 0;

    ASSERT_TRUE(dataset.size() == 4 && dataset.nFeatures() == 4);
    ASSERT_TRUE(!dataset.hasLabel());

    for (int i = 0; i != dataset.size(); ++i) {
        auto element = dataset.getItem(i);
        ASSERT_TRUE(trueData.block(i, 0, 1, trueData.cols()).isApprox(element.first));
        ASSERT_TRUE(element.second.size() == 0);
    }
}

TEST(CsvDatasetLoader, BatchWithLabels) {
    CsvDataset dataset(std::string(TEST_SOURCE_DIR) + "data.csv", true, 1);
    CsvDatasetLoader loader1(&dataset, 1);
    CsvDatasetLoader loader2(&dataset, 2);
    CsvDatasetLoader loader3(&dataset, 3);
    Tensor<float> trueData;
    Tensor<int> trueLabels;
    trueData.resize(4, 3);
    trueLabels.resize(4, 1);
    trueData << 3.5, 4, 2.2,
            7.592, 8.3, 1.1,
            2.9, 4.4444, 2.53,
            1.592, 2.8, 3.9;
    trueLabels << 1, 0, 1, 0;

    for (int i = 0; i != 10; ++i) {
        int idx = i % dataset.size();
        auto trueElement = dataset.getItem(idx);
        ASSERT_TRUE(loader1.nextBatchIndex() == (i % loader1.size()));
        auto batch = loader1.nextBatch();
        ASSERT_TRUE(trueData.block(idx, 0, 1, trueData.cols()).isApprox(batch.first));
        ASSERT_TRUE(trueLabels.block(idx, 0, 1, trueLabels.cols()).isApprox(batch.second));
    }

    for (int i = 0; i != 100000; ++i) {
        int idx = (i % (dataset.size() / 2)) * 2;
        auto trueElement = dataset.getItem(idx);
        ASSERT_TRUE(loader2.nextBatchIndex() == (i % loader2.size()));
        auto batch = loader2.nextBatch();
        ASSERT_TRUE(trueData.block(idx, 0, 2, trueData.cols()).isApprox(batch.first));
        ASSERT_TRUE(trueLabels.block(idx, 0, 2, trueLabels.cols()).isApprox(batch.second));
    }

    for (int i = 0; i != 100000; ++i) {
        int idx = (i % (dataset.size() / 3)) * 3;
        auto trueElement = dataset.getItem(idx);
        ASSERT_TRUE(loader3.nextBatchIndex() == (i % loader3.size()));
        auto batch = loader3.nextBatch();
        ASSERT_TRUE(trueData.block(idx, 0, 3, trueData.cols()).isApprox(batch.first));
        ASSERT_TRUE(trueLabels.block(idx, 0, 3, trueLabels.cols()).isApprox(batch.second));
    }
}

TEST(CsvDatasetLoader, BatchWithoutLabels) {
    CsvDataset dataset(std::string(TEST_SOURCE_DIR) + "data.csv", false, 1);
    CsvDatasetLoader loader1(&dataset, 1);
    CsvDatasetLoader loader2(&dataset, 2);
    CsvDatasetLoader loader3(&dataset, 3);
    Tensor<float> trueData;
    trueData.resize(4, 4);
    trueData << 3.5, 4, 2.2, 1,
            7.592, 8.3, 1.1, 0,
            2.9, 4.4444, 2.53, 1,
            1.592, 2.8, 3.9, 0;

    for (int i = 0; i != 10; ++i) {
        int idx = i % dataset.size();
        auto trueElement = dataset.getItem(idx);
        ASSERT_TRUE(loader1.nextBatchIndex() == (i % loader1.size()));
        auto batch = loader1.nextBatch();
        ASSERT_TRUE(trueData.block(idx, 0, 1, trueData.cols()).isApprox(batch.first));
        ASSERT_TRUE(batch.second.size() == 0);
    }

    for (int i = 0; i != 10; ++i) {
        int idx = (i % (dataset.size() / 2)) * 2;
        auto trueElement = dataset.getItem(idx);
        ASSERT_TRUE(loader2.nextBatchIndex() == (i % loader2.size()));
        auto batch = loader2.nextBatch();
        ASSERT_TRUE(trueData.block(idx, 0, 2, trueData.cols()).isApprox(batch.first));
        ASSERT_TRUE(batch.second.size() == 0);
    }

    for (int i = 0; i != 10; ++i) {
        int idx = (i % (dataset.size() / 3)) * 3;
        auto trueElement = dataset.getItem(idx);
        ASSERT_TRUE(loader3.nextBatchIndex() == (i % loader3.size()));
        auto batch = loader3.nextBatch();
        ASSERT_TRUE(trueData.block(idx, 0, 3, trueData.cols()).isApprox(batch.first));
        ASSERT_TRUE(batch.second.size() == 0);
    }
}