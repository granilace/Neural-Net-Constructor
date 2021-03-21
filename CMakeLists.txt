cmake_minimum_required(VERSION 3.16)
project(Neural_Net_Constructor)
set(CMAKE_CXX_STANDARD 17)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_subdirectory(src)

enable_testing()

add_subdirectory(test)

target_compile_definitions(unit_tests PUBLIC TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/")