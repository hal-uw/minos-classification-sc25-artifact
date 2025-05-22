#!/bin/bash
SIF_IMAGE=docker://austinguish259/sgemm:latest
NUM_KERN=100
DEVICE_ID=0
SIZE=25536
DATETIME=$(date '+%Y-%m-%d_%H-%M-%S')
echo "Number of kernels: ${NUM_KERN}"
echo "GPU ID: ${DEVICE_ID}"
echo "Matrix size: ${SIZE}"

apptainer run --nv $SIF_IMAGE make clean
apptainer run --nv $SIF_IMAGE make all
apptainer run --nv $SIF_IMAGE ./gen_data ${SIZE}
apptainer run --nv $SIF_IMAGE ncu -f ./sgemm_nvidia --target-processes all --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed --csv --devices 0 --nvtx ./sgemm_nvidia ${SIZE} ${NUM_KERN} ${DEVICE_ID} > sgemm.csv 2> sgemm_error.log 