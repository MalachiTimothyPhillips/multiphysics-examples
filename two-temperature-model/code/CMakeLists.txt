cmake_minimum_required(VERSION 3.20)
project(two_temp_tpetra LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(two_temp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

target_link_libraries(two_temp PRIVATE demo_common)

install(TARGETS two_temp
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)