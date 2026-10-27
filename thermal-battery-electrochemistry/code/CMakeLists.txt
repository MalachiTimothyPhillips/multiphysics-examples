cmake_minimum_required(VERSION 3.20)
project(thermal_battery_electrochemistry LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(thermal_battery_electrochemistry
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

target_link_libraries(thermal_battery_electrochemistry PRIVATE demo_common)

install(TARGETS thermal_battery_electrochemistry
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
