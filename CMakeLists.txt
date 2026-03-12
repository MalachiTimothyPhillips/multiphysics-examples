cmake_minimum_required(VERSION 3.20)
project(multiphysics_examples LANGUAGES CXX)

# ---- Options ----
option(BUILD_PNP            "Build poisson-nernst-planck executable" ON)
option(BUILD_THERMOELASTIC  "Build thermo-elastic executable"        ON)
option(BUILD_TWOTEMP        "Build two-temperature executable"       ON)

# ---- Common code (library) ----
add_subdirectory(common/code)

# ---- Products ----
if(BUILD_PNP)
  add_subdirectory(poisson-nernst-planck/code)
endif()

if(BUILD_THERMOELASTIC)
  add_subdirectory(thermo-elastic-model/code)
endif()

if(BUILD_TWOTEMP)
  add_subdirectory(two-temperature-model/code)
endif()