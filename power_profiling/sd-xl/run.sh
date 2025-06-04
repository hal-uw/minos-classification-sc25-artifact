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

app_name="sd-xl"
hostname=$(hostname)
prefix="metric_${app_name}"
# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
echo "Changed working directory to: $SCRIPT_DIR"

# Set application command
APP_CMD="python3 run.py"

ROCPROF_SOURCE="$SCRIPT_DIR/../../rocprofwrap"

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

# Run profiling with error checking
echo "Starting SD-XL profiling with ROCProfiler..."

# get the host name
# rename the prefix to metrics_hostname_sdxl
# set the running sets
# set the counter files
declare -A COUNTER_FILES=(
    ["set1"]="$SCRIPT_DIR/set1.json"
)


# loop through the counter files
for set_name in "${!COUNTER_FILES[@]}"; do
    COUNTER_FILE="${COUNTER_FILES[$set_name]}"
    echo "Running profiling with counter file: $COUNTER_FILE"
    
    python3 $WRAPPER --cmd "$APP_CMD" --gpus 0 --prefix "${prefix}" --counters_file "$COUNTER_FILE"
    
    if [ $? -ne 0 ]; then
        echo "Error: Profiling failed for $set_name"
        continue
    else
        echo "Successfully completed profiling for $set_name"
    fi
done

echo "All profiling runs completed."