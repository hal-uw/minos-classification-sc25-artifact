#!/bin/bash

# Environment variables
SIF_IMAGE="docker://austinguish259/llama2:latest"
CURRENT_DIR=$(pwd)

# Nsight Compute profiling command (CSV mode)
NCU_BASE="/opt/conda/nsight-compute/2022.3.0/ncu \
  --target-processes all \
  --profile-from-start no -f \
  --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
  --csv  --devices 0"

# LLaMA2 fine-tuning command
LLAMA_CMD="tune run lora_finetune_single_device --config ./config.yaml epochs=1"

# Run with Nsight Compute inside container and capture clean CSV
apptainer exec --nv $SIF_IMAGE \
  bash -c "$NCU_BASE -o llama2_ft.csv $LLAMA_CMD > llama2_output.log 2> llama2_error.log"

echo "LLaMA2 fine-tuning completed. Results:"
echo "  • CSV profiling: llama2_ft.csv"
echo "  • stdout: llama2_output.log"
echo "  • stderr: llama2_error.log"
