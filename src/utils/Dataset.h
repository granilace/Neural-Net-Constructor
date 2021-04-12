#pragma once
#include <string>
#include "Tensor.h"
#include <utility>

class CsvDataset {
 public:
    // withLabel == true if last column of csv is label. In other case all data will be saved to features
    explicit CsvDataset(const std::string& csvPath, bool withLabel, int skipRows = 0, char sep = ',');

    int size() const { return data.dimension(0); }
    int nFeatures() const {return data.dimension(1); }
    bool hasLabel() const { return withLabel; }

    // returns pair of object and its label. If hasLabel() == false then label.size() == 0
    std::pair<Tensor<float, 1>, Tensor<int, 1>> getItem(int index) const;

    ~CsvDataset();

 private:
    Tensor<float, 2> data;
    Tensor<int, 2> labels;
    bool withLabel;

    int calculateElementsAmount(const std::string& csvLine, char sep) const;
    void addElement(const std::string& csvLine, char sep, int nElements);
};
