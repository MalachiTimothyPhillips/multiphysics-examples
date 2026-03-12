#!/bin/bash

spack load mpi
spack load trilinos

rm -rf build/

cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=gold" \
  -DCMAKE_INSTALL_PREFIX=$PWD/install

cmake --build build -j
cmake --install build
