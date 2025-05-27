#!/bin/bash

# Environment variables
SIF_IMAGE="docker://austinguish259/lammps:latest"
CURRENT_DIR=$(pwd)

# Nsight Compute profiling command (CSV)
NCU_BASE="/opt/conda/nsight-compute/2022.3.0/ncu \
  --target-processes all -f \
  --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
  --csv --devices 0"

# Construct LAMMPS command
LAMMPS_CMD="../src/lmp_kokkos_cuda_mpi \
  -k on g 1 device 0 -sf kk \
  -pk kokkos neigh half neigh/qeq full newton on \
  -v x 16 -v y 8 -v z 12 -in in.reaxc.hns -nocite"

# Run with Apptainer and redirect outputs
apptainer exec --nv $SIF_IMAGE \
  bash -c "$NCU_BASE $LAMMPS_CMD > lammps.csv 2> lammps_err.log"

echo "LAMMPS run completed. Profiling CSV: lammps.csv | Errors: lammps_err.log"
