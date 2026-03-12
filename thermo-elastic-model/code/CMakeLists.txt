cmake_minimum_required(VERSION 3.20)
project(thermo_elastic_tpetra LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option A: add the common library via add_subdirectory
# (adjust the relative path if needed)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../common/code
                 ${CMAKE_BINARY_DIR}/common_build)

add_executable(thermo_elastic
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

target_link_libraries(thermo_elastic PRIVATE demo_common)