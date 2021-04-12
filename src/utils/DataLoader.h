#pragma once
#include "Tensor.h"
#include "Dataset.h"
#include <memory>

class CsvDatasetLoader {
 public:
    // if dataset.size() % batchSize != 0 then the last couple of objects will be thrown away
    CsvDatasetLoader(CsvDataset *dataset, int batchSize);
    // total amount of batches
    int size() const { return sz; }
    // returns the number of the next batch. This number lies in [0, loader.size() - 1]
    int nextBatchIndex() const { return nextBatchIdx; }

    // returns pair of objects and its labels. If dataset.hasLabel() == false then labels.size() == 0
    std::pair<Tensor<float, 2>, Tensor<int, 2>> nextBatch();

    ~CsvDatasetLoader() { dataset = nullptr; }

 private:
    CsvDataset *dataset;
    int batchSize;
    int sz;
    int nextBatchIdx;
};