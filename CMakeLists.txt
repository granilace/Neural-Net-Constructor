cmake_minimum_required(VERSION 3.17)
project(Neural_Net_Constructor1)

set(CMAKE_CXX_STANDARD 14)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(Neural_Net_Constructor1
        main.cpp src/layers/Linear.cpp src/layers/Layer.cpp src/utils/Parameter.cpp src/optimizers/Optimizer.h src/utils/Tensor.cpp src/utils/Tensor.h src/utils/Parameter.h src/layers/Layer.h src/layers/Linear.h src/layers/Sequential.cpp src/layers/Sequential.h src/utils/FloatTensor.cpp src/utils/FloatTensor.h)
target_link_libraries (Neural_Net_Constructor1 Eigen3::Eigen)







