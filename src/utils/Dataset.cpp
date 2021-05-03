#include "Tensor.h"
#include "Dataset.h"
#include <dirent.h>
#include <string>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>
#include <sys/stat.h>
#include <map>


Tensor<float, 3> read_png_file(const char *filename) {
    int width, height;
    png_byte color_type;
    png_byte bit_depth;
    png_bytep *row_pointers = NULL;

    // std::cout << "read_png_file(" << filename << ")" << std::endl;
    FILE *fp = fopen(filename, "rb");

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        throw std::runtime_error("cant create read struct for png");
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        throw std::runtime_error("cant create info struct for png");
    }

    if(setjmp(png_jmpbuf(png))) {
        throw std::runtime_error("cant set jmp buf");
    }

    png_init_io(png, fp);

    png_read_info(png, info);

    width      = png_get_image_width(png, info);
    height     = png_get_image_height(png, info);
    color_type = png_get_color_type(png, info);
    bit_depth  = png_get_bit_depth(png, info);

    // Read any color_type into 8bit depth, RGBA format.
    // See http://www.libpng.org/pub/png/libpng-manual.txt

    if(bit_depth == 16)
        png_set_strip_16(png);

    if(color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);

    // PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth.
    if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);

    if(png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);

    // These color_type don't have an alpha channel then fill it with 0xff.
    if(color_type == PNG_COLOR_TYPE_RGB ||
       color_type == PNG_COLOR_TYPE_GRAY ||
       color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

    if(color_type == PNG_COLOR_TYPE_GRAY ||
       color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    if (row_pointers) {
        throw std::runtime_error("Invalid row pointers");
    }

    row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for(int y = 0; y < height; y++) {
        row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png,info));
    }

    png_read_image(png, row_pointers);

    fclose(fp);

    Tensor<float, 3> rgb_image;
    rgb_image.resize(height, width, 3);
    for(int y = 0; y < height; y++) {
        png_bytep row = row_pointers[y];
        for(int x = 0; x < width; x++) {
            png_bytep px = &(row[x * 4]);
            float r = (float)px[0], g = (float)px[1], b = (float)px[2];
            // std::cout << '[' << y << ',' << x << "] " << "Setting (" << r << "," << g << "," << b << ")" << std::endl;
            rgb_image(x, y, 0) = r;
            rgb_image(x, y, 1) = g;
            rgb_image(x, y, 2) = b;
        }
    }
    // std::cout << "hmm " << rgb_image(0, 0, 0) << std::endl;
    //std::cout << rgb_image << std::endl;

    png_destroy_read_struct(&png, &info, NULL);
    free(row_pointers);

    // std::cout << "Successfully read " << std::endl;

    return rgb_image;
}

int calculateElementsAmount(const std::string& csvLine, char sep) {
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

template <class Predicate>
std::vector<std::string> Split(const std::string &string, Predicate is_delimiter) {
    std::vector<std::string> splitted_parts;

    auto part_begin = string.begin();
    std::string::const_iterator part_end;

    while ((part_end = std::find_if(part_begin, string.end(), is_delimiter)) != string.end()) {
        splitted_parts.emplace_back(part_begin, part_end);
        part_begin = std::next(part_end);
    }

    splitted_parts.emplace_back(part_begin, part_end);

    return splitted_parts;
}

ImageDataset::ImageDataset(const std::string &imagesDirectoryPath, const std::optional<std::string> &labelsCsvPath, char sep) {
    DIR *dir; struct dirent *diread;
    struct stat buffer;
    char path[1024];
    std::vector<std::string> imgs_paths;

    if ((dir = opendir(imagesDirectoryPath.c_str())) != nullptr) {
        while ((diread = readdir(dir)) != nullptr) {
            sprintf(path, "%s/%s", imagesDirectoryPath.c_str(), diread->d_name);
            lstat(path, &buffer);
            if (S_ISREG(buffer.st_mode)) {
                // std::cout << "appending path: " << path << std::endl;
                imgs_paths.emplace_back(path);
            }
        }
        closedir(dir);
    } else {
        throw std::runtime_error("opendir");
    }
    std::sort(imgs_paths.begin(), imgs_paths.end());

    auto img_example = read_png_file((imgs_paths[0]).c_str());
    auto img_example_dimensions = img_example.dimensions();
    auto height = img_example_dimensions[0], width = img_example_dimensions[1], channels = img_example_dimensions[2];

    auto imgs_count = imgs_paths.size();
    data.resize(imgs_count, height, width, channels);
    auto tmp = data.dimensions();
    // std::cout << "hmm6 " << tmp[0] << " " << tmp[1] << " " << tmp[2] << " " << tmp[3] << std::endl;

    for (size_t img_idx = 0; img_idx < imgs_count; ++img_idx) {
        auto img = read_png_file(imgs_paths[img_idx].c_str());
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                for (size_t channel = 0; channel < channels; ++channel) {
                    data(img_idx, y, x, channel) = img(y, x, channel);
                }
            }
        }
    }

    labels.resize(imgs_count, 1);
    if (labelsCsvPath.has_value()) {
        std::cout << "reading labels from " << labelsCsvPath.value() << std::endl;
        std::ifstream labelsCsv;
        labelsCsv.open(labelsCsvPath.value());

        std::string line;
        getline(labelsCsv, line);
        auto line_parts = Split(line, [sep](char ch) { return ch == sep; });
        assert(line_parts.size() == 2);
        int fname_idx = -1, label_idx = -1;
        if (line_parts[0] == "fname") {
            fname_idx = 0;
        } else if (line_parts[0] == "label") {
            label_idx = 1;
        }
        if (line_parts[1] == "fname") {
            fname_idx = 0;
        } else if (line_parts[1] == "label") {
            label_idx = 1;
        }
        assert(fname_idx != -1 && label_idx != -1);

        std::map<std::string, int> fpath2label;
        while (getline(labelsCsv, line)) {
            auto line_parts = Split(line, [sep](char ch) { return ch == sep; });
            assert(line_parts.size() == 2);
            fpath2label[imagesDirectoryPath + "/" + line_parts[fname_idx]] = std::stoi(line_parts[label_idx]);
        }
        labelsCsv.close();

        for (size_t img_idx = 0; img_idx < imgs_count; ++img_idx) {
            // std::cout << "Setting label " << fpath2label.at(imgs_paths[img_idx]) << " for path " << imgs_paths[img_idx] << std::endl;
            labels(img_idx) = fpath2label.at(imgs_paths[img_idx]);
        }
    } else {
        for (size_t img_idx = 0; img_idx < imgs_count; ++img_idx) {
            labels(img_idx) = -1;
        }
    }
    // std::cout << labels << std::endl;

    assert(data.dimension(0) == labels.dimension(0));
}

std::pair<Tensor<float, 3>, Tensor<int, 1>> ImageDataset::getItem(int index) const {
    return {data.chip(index, 0).eval(), labels.chip(index, 0).eval()};
}