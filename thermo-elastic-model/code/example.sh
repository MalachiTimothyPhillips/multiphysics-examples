#!/bin/bash
let Nx=256
let Ny=256

mpirun -n 8 build/thermo_elastic --Nx=$Nx --Ny=$Ny  --eta=4e2 --beta=1e0 --solver=ifpack2 --solver-xml=ifpack2.xml
mpirun -n 8 build/thermo_elastic --Nx=$Nx --Ny=$Ny  --eta=4e2 --beta=1e0 --solver=teko-bgs --solver-xml=teko-bgs.xml
mpirun -n 8 build/thermo_elastic --Nx=$Nx --Ny=$Ny  --eta=4e2 --beta=1e0 --solver=teko-amg --solver-xml=teko-amg.xml
