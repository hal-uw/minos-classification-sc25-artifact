CURRENT_DIR=$(pwd)
export TRANSFORMERS_CACHE=$CURRENT_DIR/../../datasets/model_cache/data
export HF_HOME=$CURRENT_DIR/../../datasets/model_cache
############ todo  need to create a new apptainer with ncu installed!!!!!!
SIF_IMAGE="docker://austinguish259/vllm_ncu:latest"
NCU="/opt/nvidia/nsight-compute/2025.1.1/ncu -o llama3_inf.csv -f --target-processes all  --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed --csv --devices 0 "
CMD="timeout -s SIGINT 24h $NCU python3 off.py --output_len 5"

apptainer exec --nv $SIF_IMAGE bash -c "$CMD"