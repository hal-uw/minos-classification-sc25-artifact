SIF_IMAGE="docker://austinguish259/training_gnn:latest"
# GET CURRENT PATH
CURRENT_DIR=$(pwd)
SCRIPT_PATH="$CURRENT_DIR/train_rgnn_multi_gpu.py"
DATA_PATH="$CURRENT_DIR/../../datasets/GNN"
MODEL="rgat"
DATASET_SIZE="tiny"
LAYOUT="CSC"

NCU="/opt/conda/nsight-compute/2022.3.0/ncu --profile-from-start no --target-processes all -f -o gnn.csv --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed --csv --devices 0 "

# Execute the Apptainer command with the defined variables
# apptainer exec --nv $SIF_IMAGE $NCU_PATH --target-processes all --section AppClass --csv --profile-from-start no --nvtx python $SCRIPT_PATH --model=$MODEL --dataset_size=$DATASET_SIZE --layout=$LAYOUT --path=$DATA_PATH --warmup_iters=1 --running_iters=2
apptainer exec --nv $SIF_IMAGE $NCU python $SCRIPT_PATH --model=$MODEL --dataset_size=$DATASET_SIZE --layout=$LAYOUT --path=$DATA_PATH --running_iters=2