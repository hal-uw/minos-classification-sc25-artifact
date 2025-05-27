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
NCU_BASE="/opt/conda/nsight-compute/2022.3.0/ncu \
  --nvtx \
  --profile-from-start no \
  --target-processes all -f \
  --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
  --csv --devices 0"

# Wrapper function to run NCU profiling and save to CSV
run_test () {
  local output_csv=$1
  local binary=$2
  local matrix=$3

  echo "Running $(basename $binary) on $(basename $matrix)..."
  apptainer exec --nv $SIF_IMAGE \
    bash -c "$NCU_BASE $binary -m $matrix > $output_csv 2> ${output_csv%.csv}_err.log"
}

# PageRank tests
run_test pr_indochina.csv $PR_BIN_PATH $INDOCHINA_DATA_PATH
run_test pr_att.csv       $PR_BIN_PATH $ATT_DATA_PATH

# BFS tests
run_test bfs_indochina.csv $BFS_BIN_PATH $INDOCHINA_DATA_PATH
run_test bfs_kron.csv      $BFS_BIN_PATH $KRON_DATA_PATH

# SSSP tests
run_test sssp_kron.csv      $SSSP_BIN_PATH $KRON_DATA_PATH
run_test sssp_indochina.csv $SSSP_BIN_PATH $INDOCHINA_DATA_PATH

# BC tests
run_test bc_kron.csv      $BC_BIN_PATH $KRON_DATA_PATH
run_test bc_indochina.csv $BC_BIN_PATH $INDOCHINA_DATA_PATH

echo "All tests completed."
