#!/bin/bash

# Environment variables
SIF_IMAGE="docker://austinguish259/m-psdns:latest"
CURRENT_DIR=$(pwd)
NCU_BASE="ncu --target-processes all -f --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed --csv --devices 0"
$NCU_BASE -o m-psdns.csv DNS_PEN_GPU_p4.x -i $CURRENT_DIR/../../datasets/m-psdns/input

echo "m-psdns run completed