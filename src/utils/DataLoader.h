#pragma once
#include "Tensor.h"
#include "Dataset.h"
#include <memory>

class CsvDatasetLoader {
 public:
    CsvDatasetLoader(CsvDataset *dataset, int batchSize);
    int size() const { return sz; }
    // returns the number of the next batch. This number lies in [0, loader.size() - 1]
    int nextBatchIndex() const { return nextBatchIdx; }

    // returns pair of objects and its labels. If dataset.hasLabel() == false then labels.size() == 0
    std::pair<Tensor<float>, Tensor<int>> nextBatch();

    ~CsvDatasetLoader() { dataset = nullptr; }

 private:
    CsvDataset *dataset;
    int batchSize;
    int sz;
    int nextBatchIdx;
};