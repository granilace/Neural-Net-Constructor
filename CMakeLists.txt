cmake_minimum_required(VERSION 3.17)
project(Neural_Net_Constructor)
set(CMAKE_CXX_STANDARD 17)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_subdirectory(src)

enable_testing()

add_subdirectory(test)

