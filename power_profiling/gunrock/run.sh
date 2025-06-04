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

# Set default value for gpu_model if not provided
if [ -z "$gpu_model" ]; then
    gpu_model="mi210"
fi



prefix="metric"
# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
echo "Changed working directory to: $SCRIPT_DIR"

if [ "$gpu_model" = "mi300a" ]; then
    PR_BIN="/usr/bin/pr_gfx942"
    BFS_BIN="/usr/bin/bfs_gfx942"
    SSSP_BIN="/usr/bin/sssp_gfx942"
else
    PR_BIN="/usr/bin/pr_gfx90a"
    BFS_BIN="/usr/bin/bfs_gfx90a"
    SSSP_BIN="/usr/bin/sssp_gfx90a"
fi

PR_ARGS="-n 10 -m"
PR_INPUT="../../datasets/gunrock/indochina-2004/indochina-2004.mtx"

BFS_ARGS="-s 13 -m"
BFS_INPUT="../../datasets/gunrock/kron_g500-logn21/kron_g500-logn21.mtx"

SSSP_ARGS="-s 13 -m"
SSSP_INPUT="../../datasets/gunrock/kron_g500-logn21/kron_g500-logn21.mtx"

ROCPROF_SOURCE="$SCRIPT_DIR/../../rocprofwrap"

# Check if PageRank binary exists
if [ ! -f "$PR_BIN" ]; then
    echo "Error: PageRank binary not found: $PR_BIN"
    exit 1
fi

# Check if input file exists
if [ ! -f "$PR_INPUT" ]; then
    echo "Error: Input file not found: $PR_INPUT"
    exit 1
fi



if [ ! -d "$ROCPROF_SOURCE" ]; then
    echo "Error: ROCProfiler wrapper directory not found: $ROCPROF_SOURCE"
    exit 1
fi

echo "ROCPROF_SOURCE -> $ROCPROF_SOURCE"
ls -l "$ROCPROF_SOURCE/Makefile"
mkdir -p ./rocprofwrap
cp -r $ROCPROF_SOURCE ./

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

# Configure environment variables
export WRAPPER_ROOT=$SCRIPT_DIR/rocprofwrap/
export HSA_TOOLS_LIB=/opt/rocm/lib/librocprofiler64.so.1
export LD_LIBRARY_PATH=/opt/rocm/lib/:$LD_LIBRARY_PATH

# Check if ROCProfiler library exists
if [ ! -f "$HSA_TOOLS_LIB" ]; then
    echo "Error: ROCProfiler library not found: $HSA_TOOLS_LIB"
    exit 1
fi

# Set wrapper path and verify its existence
WRAPPER="$SCRIPT_DIR/rocprofwrap/rocprofwrap.py"
if [ ! -f "$WRAPPER" ]; then
    echo "Error: Wrapper script not found: $WRAPPER"
    exit 1
fi



# Define counter sets
declare -A COUNTER_FILES=(
    ["set1"]="$SCRIPT_DIR/set1.json"
)


echo "Starting PageRank profiling with ROCProfiler..."

# Loop through counter sets
for set_name in "${!COUNTER_FILES[@]}"; do
    COUNTER_FILE="${COUNTER_FILES[$set_name]}"
    
    # Check if counter file exists
    if [ ! -f "$COUNTER_FILE" ]; then
        echo "Warning: Counter file not found: $COUNTER_FILE"
        continue
    fi
    
    echo "Running profiling with counter file: $COUNTER_FILE"
    
    python3 $WRAPPER --cmd "$PR_BIN $PR_ARGS $PR_INPUT" \
                    --gpus 0 \
                    --prefix "${prefix}_pagerank_indochina" \
                    --counters_file "$COUNTER_FILE"

    # test bfs and sssp on kron_g500-logn21.mtx

    python3 $WRAPPER --cmd "$BFS_BIN $BFS_ARGS $BFS_INPUT" \
                    --gpus 0 \
                    --prefix "${prefix}_bfs_kron" \
                    --counters_file "$COUNTER_FILE"

    python3 $WRAPPER --cmd "$SSSP_BIN $SSSP_ARGS $SSSP_INPUT" \
                    --gpus 0 \
                    --prefix "${prefix}_sssp_kron" \
                    --counters_file "$COUNTER_FILE"

    
    if [ $? -ne 0 ]; then
        echo "Error: Profiling failed with $set_name"
        continue
    else
        echo "Successfully completed profiling with $set_name"
    fi
done

echo "All profiling runs completed."