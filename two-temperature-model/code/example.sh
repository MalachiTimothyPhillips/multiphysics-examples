#!/bin/bash
let Nx=1024
let Ny=1024
mpirun -n 8 build/two_temp --Nx=$Nx --Ny=$Ny  --g=1e4 --solver=ifpack2 --solver-xml=ifpack2.xml
mpirun -n 8 build/two_temp --Nx=$Nx --Ny=$Ny  --g=1e4 --solver=teko-bgs --solver-xml=teko-bgs.xml
mpirun -n 8 build/two_temp --Nx=$Nx --Ny=$Ny  --g=1e4 --solver=teko-amg --solver-xml=teko-amg.xml
