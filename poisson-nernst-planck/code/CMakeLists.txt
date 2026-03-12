cmake_minimum_required(VERSION 3.20)
project(poisson_nernst_planck_tpetra LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(poisson_nernst_planck
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

target_link_libraries(poisson_nernst_planck PRIVATE demo_common)

install(TARGETS poisson_nernst_planck
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)