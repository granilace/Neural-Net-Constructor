#include "DataLoader.h"

CsvDatasetLoader::CsvDatasetLoader(CsvDataset *dataset, int batchSize) : dataset(dataset), batchSize(batchSize) {
    sz = dataset->size() / batchSize;
    curBatchIdx = 0;
}

std::pair<Tensor<float>, Tensor<int>> CsvDatasetLoader::getNextBatch() {
    Tensor<float> x;
    Tensor<int> y;
    x.resize(batchSize, dataset->nFeatures());
    if (dataset->trainMode()) {
        y.resize(batchSize, 1);
    }

    if (curBatchIdx >= sz) {
        curBatchIdx = 0;
    }

    int startIdx = curBatchIdx * batchSize;
    int endIdx = (curBatchIdx + 1) * batchSize;
    assert(endIdx <= dataset->size());

    for (int i = startIdx; i != endIdx; ++i) {
        auto curItem = dataset->getItem(i);
        x.block(i - startIdx, 0, 1, x.cols()) = curItem.first;
        if (dataset->trainMode()) {
            y.block(i - startIdx, 0, 1, 1) = curItem.second;
        }
    }
    ++curBatchIdx;
    return std::pair<Tensor<float>, Tensor<int>>(x, y);
}