#!/bin/bash

# Environment variables
SIF_IMAGE="docker://austinguish259/lammps:latest"
CURRENT_DIR=$(pwd)
NCU_BASE="ncu --target-processes all -f --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed --csv --devices 0"
CMD=$NCU_BASE -o lammps.csv ../src/lmp_kokkos_cuda_mpi -k on g 1 device 0 -sf kk -pk kokkos neigh half neigh/qeq full newton on -v x 16 -v y 8 -v z 12 -in in.reaxc.hns -nocite

apptainer exec --nv $SIF_IMAGE bash -c "$CMD"
echo "LAMMPS run completed. Output CSV can be found in ./reax_benchmark/ directory."