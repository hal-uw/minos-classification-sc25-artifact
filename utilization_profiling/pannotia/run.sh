#!/bin/bash

# Environment variables
SIF_IMAGE="docker://austinguish259/pannotia:latest"
PR_BIN_PATH="/usr/local/bin/pagerank_spmv"
CURRENT_DIR=$(pwd)
INDOCHINA_DATA_PATH="$CURRENT_DIR/../../datasets/gunrock/indochina-2004.mtx"
KRON_DATA_PATH="$CURRENT_DIR/../../datasets/gunrock/kron_g500-logn21.mtx"
ATT_DATA_PATH="$CURRENT_DIR/../../datasets/gunrock/pre2.mtx"
CO_PAPERS_DATA_PATH="$CURRENT_DIR/../../datasets/gunrock/coPapersDBLP.mtx"
# Common NCU parameters
NCU_BASE="ncu \
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
    bash -c "$NCU_BASE $binary $matrix 2 > $output_csv 2> ${output_csv%.csv}_err.log"
}

# PageRank tests
run_test pannotia_copapers.csv $PR_BIN_PATH $CO_PAPERS_DATA_PATH
run_test pannotia_att.csv       $PR_BIN_PATH $ATT_DATA_PATH


echo "All tests completed."
