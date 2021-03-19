#pragma once
#include "Tensor.h"
#include "Dataset.h"
#include <memory>

class CsvDatasetLoader {
 public:
    CsvDatasetLoader(CsvDataset *dataset, int batchSize);
    [[nodiscard]] int size() const { return sz; }
    [[nodiscard]] int currentBatchIndex() const { return curBatchIdx; }

    [[nodiscard]] std::pair<Tensor<float>, Tensor<int>> getNextBatch();

    ~CsvDatasetLoader() { dataset = nullptr; }

 private:
    CsvDataset *dataset;
    int batchSize;
    int sz;
    int curBatchIdx;
};