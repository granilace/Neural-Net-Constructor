add_library(nn_constructor optimizers/Loss.h)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

target_sources(nn_constructor
  PRIVATE
    layers/Identity.cpp
    layers/Linear.cpp
    layers/Sequential.cpp
    optimizers/Loss.cpp
    utils/Parameter.cpp
    utils/Tensor.cpp
    utils/Dataset.cpp
    utils/DataLoader.cpp
  PUBLIC
    layers/Identity.h
    layers/Linear.h
    layers/Sequential.h
    layers/Conv2d.h
    layers/Activations.h
    optimizers/Loss.h
    utils/Parameter.h
    utils/Tensor.h
    utils/Dataset.h
    utils/DataLoader.h
)

target_link_libraries(nn_constructor Eigen3::Eigen)

target_include_directories(nn_constructor
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
)

