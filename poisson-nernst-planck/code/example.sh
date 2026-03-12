#!/bin/bash

let Nx=1024
let Ny=1024

mpirun -n 8 build/poisson_nernst_planck --Nx=$Nx --Ny=$Ny --solver=ifpack2  --solver-xml=ifpack2.xml  --D=0.005 --eps=0.1 --zF=100 --alpha=1000
mpirun -n 8 build/poisson_nernst_planck --Nx=$Nx --Ny=$Ny --solver=teko-bgs --solver-xml=teko-bgs.xml --D=0.005 --eps=0.1 --zF=100 --alpha=1000
mpirun -n 8 build/poisson_nernst_planck --Nx=$Nx --Ny=$Ny --solver=teko-amg --solver-xml=teko-amg.xml --D=0.005 --eps=0.1 --zF=100 --alpha=1000
