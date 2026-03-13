cmake_minimum_required(VERSION 3.20)
project(demo_common LANGUAGES CXX)

add_library(demo_common STATIC)

target_compile_features(demo_common PUBLIC cxx_std_17)

# All common .cpp sources live here
file(GLOB COMMON_SRC CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

target_sources(demo_common PRIVATE ${COMMON_SRC})

# Export include path so downstream can do: #include "multiphys_mesh.hpp"
target_include_directories(demo_common
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Trilinos usage: prefer imported target if available, else fall back
find_package(Trilinos REQUIRED)

target_include_directories(demo_common PUBLIC
  ${Trilinos_INCLUDE_DIRS}
  ${Trilinos_TPL_INCLUDE_DIRS}
)

target_compile_definitions(demo_common PUBLIC
  ${Trilinos_DEFINITIONS}
)

target_link_libraries(demo_common PUBLIC
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
)

# If you truly need this:
target_link_libraries(demo_common PRIVATE gfortran)