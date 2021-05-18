cmake_minimum_required(VERSION 3.16)
project(Neural_Net_Constructor)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")

find_package (PNG REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package( Threads )

add_executable(main benchmark.cpp)

include_directories(../src)
target_link_directories(main PRIVATE src)
target_link_libraries(main Eigen3::Eigen)
# target_link_libraries(main nn_constructor)
target_link_libraries(main nlohmann_json::nlohmann_json)
target_link_libraries( main ${CMAKE_THREAD_LIBS_INIT} )