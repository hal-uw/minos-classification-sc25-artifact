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



app_name="lulesh"
hostname=$(hostname)
prefix="metric_${app_name}"
# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
echo "Changed working directory to: $SCRIPT_DIR"
# LULESH configuration
LULESH_BIN="/usr/bin/lulesh"
LULESH_ARG2=" -s 400 -i 10"

# Check if LULESH binary exists
if [ ! -f "$LULESH_BIN" ]; then
    echo "Error: LULESH binary not found: $LULESH_BIN"
    exit 1
fi


## Check and copy ROCProfiler wrapper

ROCPROF_SOURCE="$SCRIPT_DIR/../../rocprofwrap"

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

# Get hostname for metrics filename
# Define counter sets
declare -A COUNTER_FILES=(
    ["set1"]="$SCRIPT_DIR/set1.json"
)


echo "Starting LULESH profiling with ROCProfiler..."

# Loop through configurations

LULESH_ARGS="$LULESH_ARG2"
CONFIG_NAME="lulesh"
    
    
echo "Running LULESH with configuration: $LULESH_ARGS"
    
    # Loop through counter sets
    for set_name in "${!COUNTER_FILES[@]}"; do
        COUNTER_FILE="${COUNTER_FILES[$set_name]}"
        
        # Check if counter file exists
        if [ ! -f "$COUNTER_FILE" ]; then
            echo "Warning: Counter file not found: $COUNTER_FILE"
            continue
        fi
        
        echo "Running profiling with counter file: $COUNTER_FILE"
        
        python3 $WRAPPER --cmd "$LULESH_BIN $LULESH_ARGS" \
                        --gpus 0 \
                        --prefix "${prefix}" \
                        --counters_file "$COUNTER_FILE"
        
        if [ $? -ne 0 ]; then
            echo "Error: Profiling failed for configuration $CONFIG_NAME with $set_name"
            continue
        else
            echo "Successfully completed profiling for configuration $CONFIG_NAME with $set_name"
        fi
    done


echo "All profiling runs completed."