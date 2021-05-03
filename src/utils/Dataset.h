#pragma once
#include <optional>
#include <png.h>
#include <string>
#include "Tensor.h"
#include <utility>

int calculateElementsAmount(const std::string& csvLine, char sep);

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

    void addElement(const std::string& csvLine, char sep, int nElements);
};

class ImageDataset {
public:
    explicit ImageDataset(const std::string& imagesDirectoryPath, const std::optional<std::string>& labelsCsvPath = std::nullopt, char sep = ',');

    int size() const { return data.dimension(0); }

    int height() const { return data.dimension(2); }
    int width() const { return data.dimension(3); }
    int nChannels() const { return data.dimension(1); }

    // returns pair of object and its label. If hasLabel() == false then label.size() == 0
    std::pair<Tensor<float, 3>, Tensor<int, 1>> getItem(int index) const;

    // ~ImageDataset();

private:
    Tensor<float, 4> data;
    Tensor<int, 2> labels;
};