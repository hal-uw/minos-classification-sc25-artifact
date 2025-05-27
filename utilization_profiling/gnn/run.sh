#!/bin/bash

# Container and script paths
SIF_IMAGE="docker://austinguish259/training_gnn:latest"
CURRENT_DIR=$(pwd)
SCRIPT_PATH="$CURRENT_DIR/train_rgnn_multi_gpu.py"
DATA_PATH="$CURRENT_DIR/../../datasets/GNN"

# Model parameters
MODEL="rgat"
DATASET_SIZE="tiny"
LAYOUT="CSC"

# Nsight Compute command (CSV output only)
NCU="/opt/conda/nsight-compute/2022.3.0/ncu \
  --profile-from-start no \
  --target-processes all -f \
  --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
  --csv  --devices 0"

# Run profiling and separate all outputs
apptainer exec --nv $SIF_IMAGE \
  bash -c "$NCU python $SCRIPT_PATH --model=$MODEL --dataset_size=$DATASET_SIZE --layout=$LAYOUT --path=$DATA_PATH --running_iters=2 > gnn_output.log 2> gnn_error.log" > gnn.csv 2> ncu_error.log
