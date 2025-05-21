#!/bin/bash

# Environment variables
SIF_IMAGE="docker://austinguish259/m-psdns:latest"
CURRENT_DIR=$(pwd)

# NCU command with metrics
NCU_BASE="ncu --target-processes all --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed --csv --devices 0"

# Command to run inside container
CMD="$NCU_BASE DNS_PEN_GPU_p4.x -i $CURRENT_DIR/../../datasets/m-psdns/input > m-psdns.csv 2> m-psdns_error.log"

# Run with apptainer
apptainer exec --nv $SIF_IMAGE bash -c "$CMD" > apptainer_output.log 2> apptainer_error.log
