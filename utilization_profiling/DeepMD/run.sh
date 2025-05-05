#!/bin/bash
CURRENT_DIR=$(pwd)

SIF_IMAGE="docker://austinguish259/deepmd:latest"
NCU="/opt/nvidia/nsight-compute/2025.1.1/ncu --target-processes all -f -o deepmd.csv --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed --csv --devices 0 "
CMD="timeout -s SIGINT 24h $NCU dp train $CURRENT_DIR/../../datasets/DeePMD/water_se_e2_a.input.json"

apptainer exec --nv $SIF_IMAGE bash -c "$CMD"