#!/bin/bash

# Environment variables
SIF_IMAGE="docker://austinguish259/llama2:latest"
CURRENT_DIR=$(pwd)
NCU_BASE="ncu --target-processes all --profile-from-start no -f --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed --csv --devices 0"
$NCU_BASE -o llama2_ft.csv tune run lora_finetune_single_device --config ./config.yaml epochs=1

#echo "Llama2 fine-tuning run completed. Output CSV can be found in ./reax_benchmark/ directory."