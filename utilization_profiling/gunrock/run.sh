#!/bin/bash

# Environment variables
SIF_IMAGE="docker://austinguish259/gunrock:latest"
BFS_BIN_PATH="/workspace/gunrock/build/bin/bfs"
PR_BIN_PATH="/workspace/gunrock/build/bin/pr"
SSSP_BIN_PATH="/workspace/gunrock/build/bin/sssp"
BC_BIN_PATH="/workspace/gunrock/build/bin/bc"
CURRENT_DIR=$(pwd)
INDOCHINA_DATA_PATH="$CURRENT_DIR/../../datasets/gunrock/indochina-2004.mtx"
KRON_DATA_PATH="$CURRENT_DIR/../../datasets/gunrock/kron_g500-logn21.mtx"
ATT_DATA_PATH="$CURRENT_DIR/../../datasets/gunrock/pre2.mtx"

# Common NCU parameters
NCU_BASE="ncu --nvtx --profile-from-start no --target-processes all -f --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed --csv --devices 0"

# PageRank tests
echo "Running PageRank on Indochina dataset..."
apptainer exec --nv $SIF_IMAGE $NCU_BASE -o pr_indochina.csv $PR_BIN_PATH -m $INDOCHINA_DATA_PATH

echo "Running PageRank on ATT dataset..."
apptainer exec --nv $SIF_IMAGE $NCU_BASE -o pr_att.csv $PR_BIN_PATH -m $ATT_DATA_PATH

# BFS tests
echo "Running BFS on Indochina dataset..."
apptainer exec --nv $SIF_IMAGE $NCU_BASE -o bfs_indochina.csv $BFS_BIN_PATH -m $INDOCHINA_DATA_PATH

echo "Running BFS on Kron dataset..."
apptainer exec --nv $SIF_IMAGE $NCU_BASE -o bfs_kron.csv $BFS_BIN_PATH -m $KRON_DATA_PATH

# SSSP tests
echo "Running SSSP on Kron dataset..."
apptainer exec --nv $SIF_IMAGE $NCU_BASE -o sssp_kron.csv $SSSP_BIN_PATH -m $KRON_DATA_PATH

echo "Running SSSP on Indochina dataset..."
apptainer exec --nv $SIF_IMAGE $NCU_BASE -o sssp_indochina.csv $SSSP_BIN_PATH -m $INDOCHINA_DATA_PATH

# BC tests
echo "Running BC on Kron dataset..."
apptainer exec --nv $SIF_IMAGE $NCU_BASE -o bc_kron.csv $BC_BIN_PATH -m $KRON_DATA_PATH

echo "Running BC on Indochina dataset..."
apptainer exec --nv $SIF_IMAGE $NCU_BASE -o bc_indochina.csv $BC_BIN_PATH -m $INDOCHINA_DATA_PATH

echo "All tests completed."