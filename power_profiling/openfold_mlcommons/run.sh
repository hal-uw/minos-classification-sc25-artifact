#!/usr/bin/bash

while [[ $# -gt 0 ]]; do
    case $1 in
        -gpu_model)
            gpu_model="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Default GPU model if none provided
if [ -z "$gpu_model" ]; then
    gpu_model="mi210"
fi

app_name="openfold"
hostname=$(hostname)
prefix="metric_${app_name}"

# Locate this script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
echo "Changed working directory to: $SCRIPT_DIR"

# Path to the train.py inside the container
train_script="/workspace/openfold/train.py"

# Build the command line for OpenFold, pointing into datasets/
OPENFOLD_CMD="python $train_script \
  --training_dirpath $SCRIPT_DIR/../../datasets/OpenFold/mlcommons/run \
  --pdb_mmcif_chains_filepath $SCRIPT_DIR/../../datasets/OpenFold/mlcommons/pdb_data/pdb_mmcif/processed/chains.csv \
  --pdb_mmcif_dicts_dirpath $SCRIPT_DIR/../../datasets/OpenFold/mlcommons/pdb_data/pdb_mmcif/processed/dicts \
  --pdb_obsolete_filepath $SCRIPT_DIR/../../datasets/OpenFold/mlcommons/pdb_data/pdb_mmcif/processed/obsolete.dat \
  --pdb_alignments_dirpath $SCRIPT_DIR/../../datasets/OpenFold/mlcommons/pdb_data/open_protein_set/processed/pdb_alignments \
  --initialize_parameters_from $SCRIPT_DIR/../../datasets/OpenFold/mlcommons/mlperf_hpc_openfold_resumable_checkpoint_b518be46.pt \
  --train_max_pdb_release_date 2021-12-11 \
  --target_avg_lddt_ca_value 0.9 \
  --seed 1234567890 \
  --num_train_iters 5 \
  --log_every_iters 4 \
  --val_every_iters 8 \
  --local_batch_size 1 \
  --base_lr 1e-3 \
  --warmup_lr_init 1e-5 \
  --warmup_lr_iters 0 \
  --num_train_dataloader_workers 2 \
  --num_val_dataloader_workers 1 \
  --use_only_pdb_chain_ids 7ny6_A 7e6g_A"

# Point ROCProfiler wrappers into this workspace
export WRAPPER_ROOT=$SCRIPT_DIR/rocprofwrap/
export HSA_TOOLS_LIB=/opt/rocm/lib/librocprofiler64.so.1
export LD_LIBRARY_PATH=/opt/rocm/lib/:$LD_LIBRARY_PATH

ROCPROF_SOURCE="$SCRIPT_DIR/../../rocprofwrap"

# Verify and copy the ROCProfiler wrapper directory
if [ ! -d "$ROCPROF_SOURCE" ]; then
    echo "Error: ROCProfiler wrapper directory not found: $ROCPROF_SOURCE"
    exit 1
fi
cp -r "$ROCPROF_SOURCE" ./

# Rebuild gpuprof
echo "Rebuilding gpuprof..."
cd "$SCRIPT_DIR/rocprofwrap/"
make clean
make gpuprof_in_container
if [ $? -ne 0 ]; then
    echo "Error: Failed to rebuild gpuprof"
    exit 1
fi
cd "$SCRIPT_DIR"

# Ensure ROCProfiler library is available
if [ ! -f "$HSA_TOOLS_LIB" ]; then
    echo "Error: ROCProfiler library not found: $HSA_TOOLS_LIB"
    exit 1
fi

# Locate the wrapper script
WRAPPER="$SCRIPT_DIR/rocprofwrap/rocprofwrap.py"
if [ ! -f "$WRAPPER" ]; then
    echo "Error: Wrapper script not found: $WRAPPER"
    exit 1
fi

echo "Starting OpenFold (MLCommons) profiling with ROCProfiler..."
declare -A COUNTER_FILES=(
    ["set1"]="$SCRIPT_DIR/set1.json"
)

for set_name in "${!COUNTER_FILES[@]}"; do
    COUNTER_FILE="${COUNTER_FILES[$set_name]}"
    echo "Running profiling with counter file: $COUNTER_FILE"

    python3 "$WRAPPER" \
      --cmd "$OPENFOLD_CMD" \
      --gpus 0 \
      --prefix "${prefix}" \
      --counters_file "$COUNTER_FILE"

    if [ $? -ne 0 ]; then
        echo "Error: Profiling failed for $set_name"
        continue
    else
        echo "Successfully completed profiling for $set_name"
    fi
done

echo "All profiling runs completed."
