#!/bin/bash
CURRENT_DIR=$(pwd)

# Environment variables to be passed to the container
export OMP_NUM_THREADS=128
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export QUDA_ENABLE_GDR=1
export QUDA_MILC_HISQ_RECONSTRUCT=13
export QUDA_MILC_HISQ_RECONSTRUCT_SLOPPY=9

EXEC=/usr/local/milc/sm80/bin/su3_rhmd_hisq
SIF_IMAGE="docker://austinguish259/milc:latest"
INPUT=$CURRENT_DIR/../../datasets/MILC/benchmark.in
NCU="/opt/nvidia/nsight-compute/2025.1.1/ncu --target-processes all -f \
--metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
--csv --devices 0"

CMD="mpirun -np 1 $NCU $EXEC $INPUT > milc.csv 2> milc_error.log"

# Pass all the environment variables to the container
apptainer exec --nv \
  --env OMP_NUM_THREADS=$OMP_NUM_THREADS \
  --env OMP_PLACES=$OMP_PLACES \
  --env OMP_PROC_BIND=$OMP_PROC_BIND \
  --env QUDA_ENABLE_GDR=$QUDA_ENABLE_GDR \
  --env QUDA_MILC_HISQ_RECONSTRUCT=$QUDA_MILC_HISQ_RECONSTRUCT \
  --env QUDA_MILC_HISQ_RECONSTRUCT_SLOPPY=$QUDA_MILC_HISQ_RECONSTRUCT_SLOPPY \
  $SIF_IMAGE bash -c "$CMD" > apptainer_output.log 2>apptainer_error.log