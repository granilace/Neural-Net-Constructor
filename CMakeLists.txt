cmake_minimum_required(VERSION 3.16)
project(Neural_Net_Constructor)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O2")

find_package (PNG REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package( Threads )

add_subdirectory(src)

enable_testing()

add_subdirectory(test)
add_executable(cifar scripts/cifar.cpp)
include_directories(src)
target_link_directories(cifar PRIVATE src)
target_link_libraries(cifar Eigen3::Eigen)
target_link_libraries(cifar nn_constructor)
target_link_libraries( cifar ${CMAKE_THREAD_LIBS_INIT} )
file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

target_compile_definitions(unit_tests PUBLIC TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/")