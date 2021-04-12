#include "DataLoader.h"

CsvDatasetLoader::CsvDatasetLoader(CsvDataset *dataset, int batchSize) : dataset(dataset), batchSize(batchSize) {
    sz = dataset->size() / batchSize;
    nextBatchIdx = 0;
}

std::pair<Tensor<float, 2>, Tensor<int, 2>> CsvDatasetLoader::nextBatch() {
    Tensor<float, 2> x;
    Tensor<int, 2> y;
    x.resize(batchSize, dataset->nFeatures());
    if (dataset->hasLabel()) {
        y.resize(batchSize, 1);
    }

    int startIdx = nextBatchIdx * batchSize;
    int endIdx = (nextBatchIdx + 1) * batchSize;
    assert(endIdx <= dataset->size());

    for (int i = startIdx; i != endIdx; ++i) {
        auto curItem = dataset->getItem(i);
        x.chip(i - startIdx, 0) = curItem.first.eval();
        if (dataset->hasLabel()) {
            y.chip(i - startIdx, 0) = curItem.second.eval();
        }
    }
    ++nextBatchIdx;
    if (nextBatchIdx >= sz) {
        nextBatchIdx = 0;
    }
    return std::pair<Tensor<float, 2>, Tensor<int, 2>>(x, y);
}