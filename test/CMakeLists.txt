find_package (GTest REQUIRED)

add_executable(unit_tests
    layers.cpp
    optimizers.cpp
)

target_link_libraries(unit_tests
  PRIVATE
    nn_constructor
    GTest::GTest
    GTest::Main
)

# automatic discovery of unit tests
include(GoogleTest)
gtest_discover_tests(unit_tests
  PROPERTIES
    LABELS "unit"
  DISCOVERY_TIMEOUT  # how long to wait (in seconds) before crashing
    240
)

