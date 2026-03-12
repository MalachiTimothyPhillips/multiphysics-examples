#!/bin/bash

spack load mpi
spack load trilinos

rm -rf build/

cmake -S . -B build \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=gold"

cmake --build build -j
