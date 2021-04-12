#include "Tensor.h"
#include "Dataset.h"
#include <string>
#include <fstream>
#include <utility>

int CsvDataset::calculateElementsAmount(const std::string& csvLine, char sep) const {
    int result = 0;
    for (int i = 0; i != csvLine.length(); ++i) {
        if (csvLine[i] == sep) {
            ++result;
        }
    }
    return result + 1;
}

void CsvDataset::addElement(const std::string& csvLine, char sep, int nElements) {
    assert(nElements);

    int startPos = 0;
    int sepPos = csvLine.find(sep);
    assert(sepPos != std::string::npos || nElements == 1);
    for (int i = 0; i != nElements; ++i) {
        if (i != nElements - 1) {
            assert(sepPos > startPos);
            data(data.dimension(0) - 1, i) = std::stof(csvLine.substr(startPos, sepPos - startPos));
            startPos = sepPos + 1;
            sepPos = csvLine.find(sep, startPos);
        } else {
            assert(sepPos == std::string::npos);
            if (withLabel) {
                labels(labels.dimension(0) - 1) = std::stoi(csvLine.substr(startPos));
            } else {
                data(data.dimension(0) - 1, i) = std::stof(csvLine.substr(startPos));
            }
        }
    }
}

CsvDataset::CsvDataset(const std::string& csvPath, bool withLabel, int skipRows, char sep) : withLabel(withLabel) {
    std::ifstream csv;
    csv.open(csvPath);

    if (withLabel) {
        labels.resize(Eigen::NoChange, 1);
    }

    std::string line;
    bool columnsAmountDefined = false;
    int nElements = 0;
    int currentRow = 0;
    while(getline(csv, line)) {
        if (currentRow < skipRows) {
            ++currentRow;
            continue;
        }
        if (!columnsAmountDefined) {
            nElements = calculateElementsAmount(line, sep);
            assert(!withLabel || nElements > 1);
            if (withLabel) {
                data.resize(Eigen::NoChange, nElements - 1);
            } else {
                data.resize(Eigen::NoChange, nElements);
            }
            columnsAmountDefined = true;
        }

        data.resize(data.dimension(0) + 1, Eigen::NoChange);
        labels.resize(labels.dimension(0) + 1, Eigen::NoChange);

        addElement(line, sep, nElements);
    }
    csv.close();

    assert(data.dimension(0) == labels.dimension(0));
}

std::pair<Tensor<float, 1>, Tensor<int, 1>> CsvDataset::getItem(int index) const {
    if (withLabel) {
        return {
            data.chip(index, 0).eval(),
            labels.chip(index, 0).eval()
        };
    } else {
        return {
            data.chip(index, 0).eval(),
            labels.chip(0, 0).eval()
        };
    }
}

CsvDataset::~CsvDataset() {
    data.resize(0, 0);
    labels.resize(0, 0);
}