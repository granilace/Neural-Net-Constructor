#include "DataLoader.h"

CsvDatasetLoader::CsvDatasetLoader(CsvDataset *dataset, int batchSize) : dataset(dataset), batchSize(batchSize) {
    sz = dataset->size() / batchSize;
    nextBatchIdx = 0;
}

std::pair<Tensor<float>, Tensor<int>> CsvDatasetLoader::nextBatch() {
    Tensor<float> x;
    Tensor<int> y;
    x.resize(batchSize, dataset->nFeatures());
    if (dataset->hasLabel()) {
        y.resize(batchSize, 1);
    }

    int startIdx = nextBatchIdx * batchSize;
    int endIdx = (nextBatchIdx + 1) * batchSize;
    assert(endIdx <= dataset->size());

    for (int i = startIdx; i != endIdx; ++i) {
        auto curItem = dataset->getItem(i);
        x.block(i - startIdx, 0, 1, x.cols()) = curItem.first;
        if (dataset->hasLabel()) {
            y.block(i - startIdx, 0, 1, 1) = curItem.second;
        }
    }
    ++nextBatchIdx;
    if (nextBatchIdx >= sz) {
        nextBatchIdx = 0;
    }
    return std::pair<Tensor<float>, Tensor<int>>(x, y);
}