#!/bin/bash
CURRENT_DIR=$(pwd)


SIF_IMAGE="docker://austinguish259/openfold_ncu:latest"

NCU="/opt/nvidia/nsight-compute/2023.1.1/ncu --target-processes all -f \
--metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
--csv --devices 0"

EXEC=python /opt/openfold/train_openfold.py \
        $CURRENT_DIR/../../datasets/OpenFold/aqlab/pdb_data/pdb_mmcif/processed/mmcif_files \
        $CURRENT_DIR/../../datasets/OpenFold/aqlab/pdb_data/open_protein_set/processed/pdb_alignments \
        $CURRENT_DIR/../../datasets/OpenFold/aqlab/pdb_data/pdb_mmcif/processed/mmcif_files \
        full_output \
        2024-08-22 \
        --template_release_dates_cache_path=$CURRENT_DIR/../../datasets/OpenFold/aqlab/mmcif_cache.json \
        --precision=32 \
        --train_epoch_len 5 \
        --gpus=1 \
        --num_nodes=1 \
        --accumulate_grad_batches 0 \
        --replace_sampler_ddp=True \
        --seed=7152022 \
        --deepspeed_config_path= $CURRENT_DIR/../../datasets/OpenFold/aqlab/deepspeed_config.json \
        --checkpoint_every_epoch \
        --obsolete_pdbs_file_path= $CURRENT_DIR/../../datasets/OpenFold/pdb_data/pdb_mmcif/processed/mmcif_files/obsolete.dat \
        --train_chain_data_cache_path= $CURRENT_DIR/../../datasets/OpenFold/aqlab/chain_data_cache.json

CMD="timeout -s SIGINT 24h $NCU $EXEC > openfold.csv 2>openfold_error.log"

# Pass all the environment variables to the container
apptainer exec --nv \
  $SIF_IMAGE bash -c "$CMD" > apptainer_output.log 2>apptainer_error.log