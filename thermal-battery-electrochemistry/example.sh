#!/bin/bash

let numCells=10
let nxPerCell=20
let ny=16

mpirun -n 8 build/thermal_battery_electrochemistry --num-cells=$numCells --nx-per-cell=$nxPerCell --ny=$ny --solver=ifpack2 --solver-xml=ifpack2.xml
mpirun -n 8 build/thermal_battery_electrochemistry --num-cells=$numCells --nx-per-cell=$nxPerCell --ny=$ny --solver=teko-bgs --solver-xml=teko-bgs.xml
mpirun -n 8 build/thermal_battery_electrochemistry --num-cells=$numCells --nx-per-cell=$nxPerCell --ny=$ny --solver=teko-amg --solver-xml=teko-amg.xml
