#!/bin/bash

# Setup current working directory
CURRENT_DIR=$(pwd)

# HuggingFace cache paths
export TRANSFORMERS_CACHE="$CURRENT_DIR/../../datasets/model_cache/data"
export HF_HOME="$CURRENT_DIR/../../datasets/model_cache"

# Apptainer image with Nsight Compute 2025 installed
SIF_IMAGE="docker://austinguish259/vllm_ncu:latest"

# Nsight Compute profiling command
NCU="/opt/nvidia/nsight-compute/2025.1.1/ncu \
  --target-processes all \
  --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed \
  --csv --devices 0"

# Final command with timeout (max 24h)
CMD="timeout -s SIGINT 24h $NCU python3 off.py --output_len 5 > llama3_inf.csv 2> llama3_error.log"

# Execute inside Apptainer container
apptainer exec --nv "$SIF_IMAGE" bash -c "$CMD" > apptainer_output.log 2> apptainer_error.log

echo "LLaMA3 inference profiling done."
echo "  • CSV: llama3_inf.csv"
echo "  • stdout: apptainer_output.log"
echo "  • stderr: apptainer_error.log"