#!/usr/bin/bash
# Get the directory where this script is located
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

app_name="llama3"
hostname=$(hostname)
prefix="metric_${app_name}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
echo "Changed working directory to: $SCRIPT_DIR"

# Define batch sizes to test
batch_sizes=(1 8 32)

# Base directory path for VLLM


# Set application command - now using VLLM latency script
APP_CMD="$SCRIPT_DIR/../latency.sh -m meta-llama/Meta-Llama-3.1-8B-Instruct -g 1 -d float16 -i 2048 -o 128"

# Check and copy ROCProfiler wrapper
if [ ! -d "$ROCPROF_SOURCE" ]; then
    echo "Error: ROCProfiler wrapper directory not found: $ROCPROF_SOURCE"
    exit 1
fi
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


# Set the counter files
declare -A COUNTER_FILES=(
    ["set1"]="$SCRIPT_DIR/set1.json"
)

echo "Starting LLaMA 3 profiling with ROCProfiler..."

# Loop through batch sizes
for batch_size in "${batch_sizes[@]}"; do
    echo "Profiling with batch size: $batch_size"
    echo "----------------------------------------"
    
    # Create the full command with current batch size
    CURRENT_CMD="$APP_CMD -b $batch_size"
    
    # Loop through the counter files
    for set_name in "${!COUNTER_FILES[@]}"; do
        COUNTER_FILE="${COUNTER_FILES[$set_name]}"
        echo "Running profiling with counter file: $COUNTER_FILE"
        python3 $WRAPPER --cmd "$CURRENT_CMD" --gpus 0 \
            --prefix "${prefix}_bs${batch_size}" \
            --counters_file "$COUNTER_FILE"
            
        if [ $? -ne 0 ]; then
            echo "Error: Profiling failed for $set_name with batch size $batch_size"
            continue
        else
            echo "Successfully completed profiling for $set_name with batch size $batch_size"
        fi
    done
    
    echo "----------------------------------------"
    echo "Completed profiling for batch size: $batch_size"
    echo ""
    
    # Cool-down period between different batch sizes
    sleep 5
done

echo "All LLaMA 3 profiling runs completed."
