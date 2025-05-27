#!/bin/bash
SIF_IMAGE=docker://austinguish259/resnet:latest
apptainer run --nv $SIF_IMAGE ncu --target-processes all --target-processes all --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed --profile-from-start no --csv --nvtx python3 resnet_50.py > resnet_50.csv 2> resnet_50_error.log