#pragma once
#include "Tensor.h"
#include "Dataset.h"
#include <memory>

class CsvDatasetLoader {
 public:
    CsvDatasetLoader(CsvDataset *dataset, int batchSize);
    int size() const { return sz; }
    int currentBatchIndex() const { return curBatchIdx; }

    std::pair<Tensor<float>, Tensor<int>> nextBatch();

    ~CsvDatasetLoader() { dataset = nullptr; }

 private:
    CsvDataset *dataset;
    int batchSize;
    int sz;
    int curBatchIdx;
};