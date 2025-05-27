#!/bin/bash
CURRENT_DIR=$(pwd)

SIF_IMAGE="docker://austinguish259/qmcpack_cuda:latest"

NCU="/opt/nvidia/nsight-compute/2025.1.1/ncu --target-processes all -f \
--metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
--csv --devices 0"

CMD="$NCU  /usr/local/qmcpack/sm80/bin/qmcpack $CURRENT_DIR/../../datasets/qmcpack/input_nv.xml> qmcpack.csv 2>qmcpack_error.log"

apptainer exec --nv $SIF_IMAGE bash -c \"$CMD\" > apptainer_output.log 2>apptainer_error.log
