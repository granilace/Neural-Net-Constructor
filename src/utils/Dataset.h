#pragma once
#include <string>
#include "Tensor.h"
#include <utility>

class CsvDataset {
 public:
    explicit CsvDataset(const std::string& csvPath, bool train, int skipRows = 0, char sep = ',');

    [[nodiscard]] int size() const { return data.rows(); }
    [[nodiscard]] int nFeatures() const {return data.cols(); }
    [[nodiscard]] bool trainMode() const { return train; }

    [[nodiscard]] std::pair<Tensor<float>, Tensor<int>>getItem(int index) const;

    ~CsvDataset();

 //private:
    Tensor<float> data;
    Tensor<int> labels;
    bool train;

    [[nodiscard]] int calculateElementsAmount(const std::string& csvLine, char sep) const;
    void addElement(const std::string& csvLine, char sep, int nElements);
};
