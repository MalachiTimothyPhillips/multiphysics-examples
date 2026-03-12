cmake_minimum_required(VERSION 3.20)
project(thermo_elastic_tpetra LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(thermo_elastic
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

target_link_libraries(thermo_elastic PRIVATE demo_common)

install(TARGETS thermo_elastic
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)