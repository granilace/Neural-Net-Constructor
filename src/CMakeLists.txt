add_library(nn_constructor)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

target_sources(nn_constructor
  PRIVATE
    layers/Identity.cpp
    layers/Sequential.cpp
    utils/Parameter.cpp
    utils/Tensor.cpp
  PUBLIC
    layers/Identity.h
    layers/Sequential.h
    utils/Parameter.h
    utils/Tensor.h
)

target_link_libraries(nn_constructor Eigen3::Eigen)

target_include_directories(nn_constructor
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
)

