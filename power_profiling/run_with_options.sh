#!/usr/bin/bash
gpu_model="mi300a"
selected_app="all"
HF_TOKEN=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -gpu_model)
      gpu_model="$2"
      shift 2
      ;;
    -selected_app)
      selected_app="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: $0 -gpu_model <model> -selected_app <app>"
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(pwd)"
ROCPROF_SOURCE="$SCRIPT_DIR/rocprofwrap"
# dataset is in the parent directory/datasets
DATASETS_DIR="$SCRIPT_DIR/../datasets"
# Function to print usage
print_usage() {
    echo "Usage: $0 -gpu_model <model> -selected_app <app>"
    echo "Available apps:"
    echo "  qmcpack       - Run QMCPACK test"
    echo "  lulesh        - Run LULESH test"
    echo "  gunrock      - Run Gunrock PageRank test"
    echo "  sd           - Run Stable Diffusion test"
    echo "  llama2       - Run Llama2 test"
    echo "  llama3       - Run Llama3 test"
    echo "  llama2_ft    - Run Llama2 finetuning test"
    echo "  deepmd       - Run DeepMD test"
    echo "  lsms         - Run LSMS test"
    echo "  openfold_mlcommons - Run OpenFold MLCommons Version test"
    echo " resnet       - Run ResNet test"
    echo " milc        - Run MILC test"
    echo "  all          - Run all tests (default)"
    exit 1
}

# Function to run QMCPACK
run_qmcpack() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting QMCPACK test..."
    QMCPACK_IMAGE=docker://austinguish259/rocm_gpu_qmcpack:latest
    QMCPACK_RUN_SCRIPT="$SCRIPT_DIR/qmcpack/run.sh"
    chmod a+x $QMCPACK_RUN_SCRIPT

    if [ ! -f "$QMCPACK_RUN_SCRIPT" ]; then
        echo "Error: QMCPACK files not found"
        return 1
    fi
    # Run with Apptainer using --env flag
    apptainer run --rocm \
        --env OMP_NUM_THREADS=$OMP_NUM_THREADS \
        --env HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES \
        --env ROCPROF_SOURCE=$ROCPROF_SOURCE \
        --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        
        $QMCPACK_IMAGE \
        bash $SCRIPT_DIR/qmcpack/run.sh -gpu_model $gpu_model
    
    return $?
}

# Function to run LULESH
run_lulesh() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting LULESH test..."
    LULESH_IMAGE=docker://austinguish259/rocm_lulesh_pr:latest
    LULESH_RUN_SCRIPT="$SCRIPT_DIR/lulesh/run.sh"
    chmod a+x $LULESH_RUN_SCRIPT

    if [ ! -f "$LULESH_RUN_SCRIPT" ]; then
        echo "Error: LULESH files not found"
        return 1
    fi

    # Run with Apptainer using --env flag
    apptainer run --rocm \
        --env HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES \
        --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        --env ROCPROF_SOURCE=$ROCPROF_SOURCE \
        
        $LULESH_IMAGE \
        bash $SCRIPT_DIR/lulesh/run.sh -gpu_model $gpu_model
    
    return $?
}

# Function to run Gunrock
run_gunrock() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting gunrock pagerank test..."
    GUNROCK_IMAGE=docker://austinguish259/rocm_lulesh_pr:latest
    GUNROCK_RUN_SCRIPT="$SCRIPT_DIR/gunrock/run.sh"
    chmod a+x $GUNROCK_RUN_SCRIPT

    if [ ! -f "$GUNROCK_RUN_SCRIPT" ]; then
        echo "Error: Gunrock files not found"
        return 1
    fi

    # Set environment variables for Apptainer
    export HIP_VISIBLE_DEVICES=0
    export CUDA_VISIBLE_DEVICES=0
    export ROCPROF_SOURCE="$ROCPROF_SOURCE"

    # Run with Apptainer using --env flag
    apptainer run --rocm \
        --env HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES \
        --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        --env ROCPROF_SOURCE=$ROCPROF_SOURCE \
        
        $GUNROCK_IMAGE \
        bash $SCRIPT_DIR/gunrock/run.sh -gpu_model $gpu_model
    
    return $?
}

# Function to run Stable Diffusion
run_sd() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Stable Diffusion test..."
    SD_IMAGE=docker://austinguish259/rocm_sd-xl-turbo:latest
    SD_RUN_SCRIPT="$SCRIPT_DIR/sd-xl/run.sh"
    CACHE_DIR=/mnt/dcgpuval/rutwjain/apps/model_cache
    HF_DATA_DIR=/mnt/dcgpuval/rutwjain/apps/model_cache/data

    mkdir -p $CACHE_DIR $HF_DATA_DIR
    chmod a+x $SD_RUN_SCRIPT

    if [ ! -f "$SD_RUN_SCRIPT" ]; then
        echo "Error: Stable Diffusion files not found"
        return 1
    fi

    # Set environment variables for Apptainer
    export TRANSFORMERS_CACHE=$CACHE_DIR
    export HF_HOME=$HF_DATA_DIR
    export HF_TOKEN="$HF_TOKEN"
    export HIP_VISIBLE_DEVICES=0
    export CUDA_VISIBLE_DEVICES=0
    export ROCPROF_SOURCE="$ROCPROF_SOURCE"

    # Run with Apptainer using --env flag
    apptainer run --rocm \
        --env TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE \
        --env HF_HOME=$HF_HOME \
        --env HF_TOKEN=$HF_TOKEN \
        --env HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES \
        --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        --env ROCPROF_SOURCE=$ROCPROF_SOURCE \
        --bind "$CACHE_DIR:$CACHE_DIR" \
        --bind "$HF_DATA_DIR:$HF_DATA_DIR" \
        
        $SD_IMAGE \
        bash $SCRIPT_DIR/sd-xl/run.sh -gpu_model $gpu_model
    
    return $?
}

# Function to run Llama2
run_llama2() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Llama2 test..."
    VLLM_IMAGE=docker://rocm/vllm-dev:20241030
    LLAMA2_RUN_SCRIPT="$SCRIPT_DIR/vllm/llama2/run.sh"
    CACHE_DIR=/mnt/dcgpuval/rutwjain/apps/model_cache/data
    HF_TOKEN="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"

    mkdir -p $CACHE_DIR
    chmod a+x $LLAMA2_RUN_SCRIPT

    if [ ! -f "$LLAMA2_RUN_SCRIPT" ]; then
        echo "Error: Llama2 files not found"
        return 1
    fi

    # Set environment variables for Apptainer
    export HUGGINGFACE_HUB_CACHE="$CACHE_DIR"
    export HF_CACHE_DIR="$CACHE_DIR"
    export MAD_SECRETS_HFTOKEN="$HF_TOKEN"
    export HF_HOME="$CACHE_DIR"
    export HF_TOKEN="$HF_TOKEN"
    export HIP_VISIBLE_DEVICES=0
    export CUDA_VISIBLE_DEVICES=0
    export ROCPROF_SOURCE="$ROCPROF_SOURCE"

    # Run with Apptainer using --env flag
    apptainer run --rocm \
        --env HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE \
        --env HF_CACHE_DIR=$HF_CACHE_DIR \
        --env MAD_SECRETS_HFTOKEN=$MAD_SECRETS_HFTOKEN \
        --env HF_HOME=$HF_HOME \
        --env HF_TOKEN=$HF_TOKEN \
        --env HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES \
        --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        --env ROCPROF_SOURCE=$ROCPROF_SOURCE \
        --bind "$CACHE_DIR:$CACHE_DIR" \
        
        $VLLM_IMAGE \
        bash $SCRIPT_DIR/vllm/llama2/run.sh -gpu_model $gpu_model
    
    return $?
}

run_mixtral() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Mixtral test..."
    VLLM_IMAGE=docker://rocm/vllm-dev:20241030
    MIXTRAL_RUN_SCRIPT="$SCRIPT_DIR/vllm/mixtral-8x7B/run.sh"
    CACHE_DIR=/mnt/dcgpuval/rutwjain/apps/model_cache/data
    HF_TOKEN="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"

    mkdir -p $CACHE_DIR
    chmod a+x $MIXTRAL_RUN_SCRIPT

    if [ ! -f "$MIXTRAL_RUN_SCRIPT" ]; then
        echo "Error: MIXTRAL files not found"
        return 1
    fi

    # Set environment variables for Apptainer
    export HUGGINGFACE_HUB_CACHE="$CACHE_DIR"
    export HF_CACHE_DIR="$CACHE_DIR"
    export MAD_SECRETS_HFTOKEN="$HF_TOKEN"
    export HF_TOKEN="$HF_TOKEN"
    export HF_HOME="$CACHE_DIR"
    export HIP_VISIBLE_DEVICES=0,1
    export ROCPROF_SOURCE="$ROCPROF_SOURCE"
    export CUDA_VISIBLE_DEVICES=0,1

    # Run with Apptainer using --env flag
    apptainer run --rocm \
        --env HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE \
        --env HF_CACHE_DIR=$HF_CACHE_DIR \
        --env MAD_SECRETS_HFTOKEN=$MAD_SECRETS_HFTOKEN \
        --env HF_TOKEN=$HF_TOKEN \
        --env HF_HOME=$HF_HOME \
        --env HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES \
        --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        --env ROCPROF_SOURCE=$ROCPROF_SOURCE \
        --bind "$CACHE_DIR:$CACHE_DIR" \
        
        $VLLM_IMAGE \
        bash $SCRIPT_DIR/vllm/mixtral-8x7B/run.sh -gpu_model $gpu_model
    
    return $?
}

run_deepseek() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting deepseek test..."
    VLLM_IMAGE=docker://rocm/vllm-dev:20241030
    DeepSeekR1_RUN_SCRIPT="$SCRIPT_DIR/vllm/deepseek-llama/run.sh"
    CACHE_DIR=/mnt/dcgpuval/rutwjain/apps/model_cache/data
    HF_TOKEN="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"

    mkdir -p $CACHE_DIR
    chmod a+x $DeepSeekR1_RUN_SCRIPT

    if [ ! -f "$DeepSeekR1_RUN_SCRIPT" ]; then
        echo "Error: deepseek files not found"
        return 1
    fi

    # Set environment variables for Apptainer
    export HUGGINGFACE_HUB_CACHE="$CACHE_DIR"
    export HF_CACHE_DIR="$CACHE_DIR"
    export MAD_SECRETS_HFTOKEN="$HF_TOKEN"
    export HF_HOME="$CACHE_DIR"
    export HF_TOKEN="$HF_TOKEN"
    export HIP_VISIBLE_DEVICES=0
    export CUDA_VISIBLE_DEVICES=0
    export ROCPROF_SOURCE="$ROCPROF_SOURCE"

    # Run with Apptainer using --env flag
    apptainer run --rocm \
        --env HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE \
        --env HF_CACHE_DIR=$HF_CACHE_DIR \
        --env MAD_SECRETS_HFTOKEN=$MAD_SECRETS_HFTOKEN \
        --env HF_HOME=$HF_HOME \
        --env HF_TOKEN=$HF_TOKEN \
        --env HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES \
        --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        --env ROCPROF_SOURCE=$ROCPROF_SOURCE \
        --bind "$CACHE_DIR:$CACHE_DIR" \
        
        $VLLM_IMAGE \
        bash $SCRIPT_DIR/vllm/deepseek-llama/run.sh -gpu_model $gpu_model
    
    return $?
}

# Function to run Llama3
run_llama3() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Llama3 test..."
    VLLM_IMAGE=docker://rocm/vllm-dev:20241030
    LLAMA3_RUN_SCRIPT="$SCRIPT_DIR/vllm/llama3/run.sh"
    CACHE_DIR=/mnt/dcgpuval/rutwjain/apps/model_cache/data
    HF_TOKEN="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"
    
    mkdir -p $CACHE_DIR
    chmod a+x $LLAMA3_RUN_SCRIPT
    
    if [ ! -f "$LLAMA3_RUN_SCRIPT" ]; then
        echo "Error: Llama3 files not found"
        return 1
    fi

    # Set environment variables for Apptainer
    export HUGGINGFACE_HUB_CACHE="$CACHE_DIR"
    export HF_CACHE_DIR="$CACHE_DIR"
    export MAD_SECRETS_HFTOKEN="$HF_TOKEN"
    export HF_TOKEN="$HF_TOKEN"
    export HF_HOME="$CACHE_DIR"
    export HIP_VISIBLE_DEVICES=0
    export CUDA_VISIBLE_DEVICES=0

    # Run with Apptainer using --env flag
    apptainer run --rocm \
        --env HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE \
        --env HF_CACHE_DIR=$HF_CACHE_DIR \
        --env MAD_SECRETS_HFTOKEN=$MAD_SECRETS_HFTOKEN \
        --env HF_TOKEN=$HF_TOKEN \
        --env HF_HOME=$HF_HOME \
        --env HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES \
        --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        --bind "$CACHE_DIR:$CACHE_DIR" \
        
        $VLLM_IMAGE \
        bash $SCRIPT_DIR/vllm/llama3/run.sh -gpu_model $gpu_model
    
    return $?
}

run_qwen() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Qwen2.5 test..."
    VLLM_IMAGE=docker://rocm/vllm-dev:20241030
    QWEN_RUN_SCRIPT="$SCRIPT_DIR/vllm/qwen2.5/run.sh"
    CACHE_DIR=/mnt/dcgpuval/rutwjain/apps/model_cache/data
    
    mkdir -p $CACHE_DIR
    chmod a+x $QWEN_RUN_SCRIPT
    
    if [ ! -f "$QWEN_RUN_SCRIPT" ]; then
        echo "Error: Qwen2.5 files not found"
        return 1
    fi

    # Set environment variables for Apptainer
    export HUGGINGFACE_HUB_CACHE="$CACHE_DIR"
    export HF_CACHE_DIR="$CACHE_DIR"
    export MAD_SECRETS_HFTOKEN="$HF_TOKEN"
    export HF_TOKEN="$HF_TOKEN"
    export HF_HOME="$CACHE_DIR"
    export HIP_VISIBLE_DEVICES=0
    export CUDA_VISIBLE_DEVICES=0
    export ROCPROF_SOURCE="$ROCPROF_SOURCE"

    # Run with Apptainer using --env flag
    apptainer run --rocm \
        --env HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE \
        --env HF_CACHE_DIR=$HF_CACHE_DIR \
        --env MAD_SECRETS_HFTOKEN=$MAD_SECRETS_HFTOKEN \
        --env HF_TOKEN=$HF_TOKEN \
        --env HF_HOME=$HF_HOME \
        --env HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES \
        --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        --env ROCPROF_SOURCE=$ROCPROF_SOURCE \
        --bind "$CACHE_DIR:$CACHE_DIR" \
        
        $VLLM_IMAGE \
        bash $SCRIPT_DIR/vllm/qwen2.5/run.sh -gpu_model $gpu_model
    
    return $?
}

# Function to run Llama2 finetuning
run_llama2_ft() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Llama2 finetuning test..."
    LLAMA2_FT_IMAGE=docker://austinguish259/torchtune:latest
    LLAMA2_FT_RUN_SCRIPT="$SCRIPT_DIR/llama2_ft/run.sh"
    CACHE_DIR=/mnt/dcgpuval/rutwjain/apps/model_cache/data
    
    chmod a+x $LLAMA2_FT_RUN_SCRIPT
    
    if [ ! -f "$LLAMA2_FT_RUN_SCRIPT" ]; then
        echo "Error: Llama2 finetuning files not found"
        return 1
    fi
    
    # Set environment variables for Apptainer
    export HIP_VISIBLE_DEVICES=0
    export CUDA_VISIBLE_DEVICES=0
    export HUGGINGFACE_HUB_CACHE="$CACHE_DIR"
    export HF_TOKEN="$HF_TOKEN"
    export ROCPROF_SOURCE="$ROCPROF_SOURCE"
    export HF_CACHE_DIR="$CACHE_DIR"

    # Run with Apptainer using --env flag
    apptainer run --rocm \
        --env HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES \
        --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        --env HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE \
        --env HF_TOKEN=$HF_TOKEN \
        --env ROCPROF_SOURCE=$ROCPROF_SOURCE \
        --env HF_CACHE_DIR=$HF_CACHE_DIR \
        
        $LLAMA2_FT_IMAGE \
        bash $SCRIPT_DIR/llama2_ft/run.sh -gpu_model $gpu_model
    
    return $?
}

run_deepmd() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting deepmd test..."
    DEEPMD_IMAGE=docker://austinguish259/rocm_deepmd:latest
    DEEPMD_RUN_SCRIPT="$SCRIPT_DIR/deepmd/run.sh"
    chmod a+x $DEEPMD_RUN_SCRIPT

    if [ ! -f "$DEEPMD_RUN_SCRIPT" ]; then
        echo "Error: DEEPMD files not found"
        return 1
    fi

    # Set environment variables for Apptainer
    export HIP_VISIBLE_DEVICES=0
    export CUDA_VISIBLE_DEVICES=0
    export ROCPROF_SOURCE="$ROCPROF_SOURCE"

    # Run with Apptainer using --env flag
    apptainer run --rocm \
        --env HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES \
        --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        --env ROCPROF_SOURCE=$ROCPROF_SOURCE \
        
        $DEEPMD_IMAGE \
        bash $SCRIPT_DIR/deepmd/run.sh -gpu_model $gpu_model
    
    return $?
}

run_lsms(){
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting LSMS test..."
    LSMS_IMAGE=docker://austinguish259/rocm_lsms:latest
    LSMS_RUN_SCRIPT="$SCRIPT_DIR/lsms/run.sh"
    chmod a+x $LSMS_RUN_SCRIPT
    
    if [ ! -f "$LSMS_RUN_SCRIPT" ]; then
        echo "Error: LSMS files not found"
        return 1
    fi
    
    # Set environment variables for Apptainer
    export HIP_VISIBLE_DEVICES=0
    export ROCPROF_SOURCE="$ROCPROF_SOURCE"
    export CUDA_VISIBLE_DEVICES=0

    # Run with Apptainer using --env flag
    apptainer run --rocm \
        --env HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES \
        --env ROCPROF_SOURCE=$ROCPROF_SOURCE \
        --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        
        $LSMS_IMAGE \
        bash $SCRIPT_DIR/lsms/run.sh -gpu_model $gpu_model
    
    return $? 
}

run_resnet(){
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ResNet test..."
    RESNET_IMAGE=docker://austinguish259/resnet50:latest
    RESNET_RUN_SCRIPT="$SCRIPT_DIR/resnet/run.sh"
    chmod a+x $RESNET_RUN_SCRIPT
    
    if [ ! -f "$RESNET_RUN_SCRIPT" ]; then
        echo "Error: ResNet files not found"
        return 1
    fi
    
    # Set environment variables for Apptainer
    export HIP_VISIBLE_DEVICES=0
    export ROCPROF_SOURCE="$ROCPROF_SOURCE"
    export CUDA_VISIBLE_DEVICES=0

    # Run with Apptainer using --env flag
    apptainer run --rocm \
        --env HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES \
        --env ROCPROF_SOURCE=$ROCPROF_SOURCE \
        --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        
        $RESNET_IMAGE \
        bash $SCRIPT_DIR/resnet/run.sh -gpu_model $gpu_model
    
    return $? 
}

run_milc(){
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting MILC test..."
    MILC_IMAGE=docker://austinguish259/rocm_milc:latest
    MILC_RUN_SCRIPT="$SCRIPT_DIR/milc/run.sh"
    chmod a+x $MILC_RUN_SCRIPT
    
    if [ ! -f "$MILC_RUN_SCRIPT" ]; then
        echo "Error: MILC files not found"
        return 1
    fi
    
    # Set environment variables for Apptainer

    # Run with Apptainer using --env flag
    apptainer run --rocm \
        --env HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES \
        --env ROCPROF_SOURCE=$ROCPROF_SOURCE \
        --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        --env QUDA_RESOURCE_PATH="$SCRIPT_DIR/milc/qudatune" \
        --env QUDA_ENABLE_GDR=1 \
        --env  QUDA_MILC_HISQ_RECONSTRUCT=13 \
        --env  QUDA_MILC_HISQ_RECONSTRUCT_SLOPPY=9 \
        
        $MILC_IMAGE \
        bash $SCRIPT_DIR/milc/run.sh -gpu_model $gpu_model
    
    return $? 
}

run_openfold_mlcommons(){
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting OpenFold MLCommons Version test..."
    OPENFOLD_IMAGE=docker://austinguish259/rocm_openfold_mlcommons:latest
    OPENFOLD_RUN_SCRIPT="$SCRIPT_DIR/openfold_mlcommons/run.sh"
    chmod a+x $OPENFOLD_RUN_SCRIPT
    
    if [ ! -f "$OPENFOLD_RUN_SCRIPT" ]; then
        echo "Error: OpenFold MLCommons files not found"
        return 1
    fi
    
    # Set environment variables for Apptainer
    export HIP_VISIBLE_DEVICES=0
    export CUDA_VISIBLE_DEVICES=0
    export ROCPROF_SOURCE="$ROCPROF_SOURCE"

    # Run with Apptainer using --env flag
    apptainer run --rocm \
        --env HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES \
        --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        --env ROCPROF_SOURCE=$ROCPROF_SOURCE \
        
        $OPENFOLD_IMAGE \
        bash $SCRIPT_DIR/openfold_mlcommons/run.sh -gpu_model $gpu_model
    
    return $? 
}

if [ ! -z "$APP" ]; then
    selected_app=$APP
fi
if [ ! -z "$GPU_MODEL" ]; then
    gpu_model=$GPU_MODEL
fi

# check if the HF_TOKEN is set if not set, then report error

if [ -z "$HF_TOKEN" ];
then
    echo "Error: HF_TOKEN is not set. Please set it before running the script."
    exit 1
fi


case $selected_app in
    "qmcpack")
        run_qmcpack
        ;;
    "lulesh")
        run_lulesh
        ;;
    "gunrock")
        run_gunrock
        ;;
    "sd")
        run_sd
        ;;
    "llama2")
        run_llama2
        ;;
    "llama3")
        run_llama3
        ;;
    "mixtral")
        run_mixtral
        ;;
    "qwen2.5")
        run_qwen
        ;;
    "resnet")
        run_resnet
        ;;
    "milc")
        run_milc
        ;;
    "llama2_ft")
        run_llama2_ft
        ;;
    "deepmd")
        run_deepmd
        ;;
    "lsms")
        run_lsms
        ;;
    "deepseek_r1_llama8b")
        run_deepseek
        ;;
    "openfold_mlcommons")
        run_openfold_mlcommons
        ;;
    "all" | "")
        run_qmcpack || exit 1
        run_lulesh || exit 1
        run_gunrock || exit 1
        run_sd || exit 1
        run_llama2 || exit 1
        run_llama3 || exit 1
        # run_deepseek || exit 1
        # run_mixtral || exit 1
        # run_qwen || exit 1
        run_llama2_ft || exit 1
        run_deepmd || exit 1
        run_lsms || exit 1
        run_resnet || exit 1
        run_milc || exit 1
        run_openfold_mlcommons || exit 1
        ;;
    *)
        echo "Error: Invalid app name: $selected_app"
        print_usage
        ;;
esac

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All requested tests completed."