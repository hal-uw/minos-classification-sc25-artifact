#!/bin/bash

# Parse command line arguments
# --option: Optional parameter to specify which datasets to download (comma-separated)
# Example: ./script.sh --option qmcpack,gunrock,openfold_aqlab
# Default: Download all datasets
HF_TOKEN=""
ALL_DATASETS=("qmcpack" "gunrock" "openfold_aqlab" "openfold_mlcommons" "gnn" "llama2_ft")
SELECTED_DATASETS=()

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --option)
      # Split the comma-separated list into an array
      IFS=',' read -ra SELECTED_DATASETS <<< "$2"
      shift 2
      ;;
    *)
      # Unknown option
      echo "Unknown option: $1"
      echo "Usage: $0 [--option dataset1,dataset2,...]"
      echo "Available datasets: qmcpack, gunrock, openfold_aqlab, openfold_mlcommons, gnn"
      exit 1
      ;;
  esac
done

# If no datasets are specified, use all datasets
if [ ${#SELECTED_DATASETS[@]} -eq 0 ]; then
  SELECTED_DATASETS=("${ALL_DATASETS[@]}")
fi

# Function to check if a dataset should be downloaded
should_download() {
  local dataset="$1"
  for selected in "${SELECTED_DATASETS[@]}"; do
    if [[ "$selected" == "$dataset" ]]; then
      return 0  # True
    fi
  done
  return 1  # False
}

# Create datasets directory
mkdir -p datasets
cd datasets

# QMCPack
if should_download "qmcpack"; then
  echo "Downloading QMCPack datasets..."
  mkdir -p qmcpack
  cd qmcpack
  # https://github.com/QMCPACK/qmcpack/tree/0a0f0032ef579510a444a84060b19b5bd8e4a981/tests/performance/NiO
  # NiO S1 Orbital File
  curl -L -O -J https://anl.box.com/shared/static/uduxhujxkm1st8pau9muin255cxr2blb.h5
  # NiO S64 Orbital File
  curl -L -O -J https://anl.box.com/shared/static/yneul9l7rq2ad35vkt4mgmr2ijxt5vb6.h5
  cd ..
else
  echo "Skipping QMCPack datasets"
fi

# Gunrock
if should_download "gunrock"; then
  echo "Downloading Gunrock datasets..."
  mkdir -p gunrock
  cd gunrock
  
  # indochina
  curl -L -O -J https://www.cise.ufl.edu/research/sparse/MM/LAW/indochina-2004.tar.gz
  tar -xvf indochina-2004.tar.gz
  rm indochina-2004.tar.gz
  
  # att
  curl -L -O -J https://suitesparse-collection-website.herokuapp.com/MM/ATandT/pre2.tar.gz
  tar -xvf pre2.tar.gz
  rm pre2.tar.gz
  
  # coPapersDBLP
  curl -L -O -J https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/coPapersDBLP.tar.gz
  tar -xvf coPapersDBLP.tar.gz
  rm coPapersDBLP.tar.gz
  
  # kron
  curl -L -O -J https://www.cise.ufl.edu/research/sparse/MM/DIMACS10/kron_g500-logn21.tar.gz
  tar -xvf kron_g500-logn21.tar.gz
  rm kron_g500-logn21.tar.gz
  cd ..
else
  echo "Skipping Gunrock datasets"
fi

# Create OpenFold directory if any OpenFold dataset is selected
if should_download "openfold_aqlab" || should_download "openfold_mlcommons"; then
  mkdir -p OpenFold
  cd OpenFold
else
  echo "Skipping all OpenFold datasets"
fi

# OpenFold AQLab (requires mmseqs)
if should_download "openfold_aqlab"; then
  echo "Downloading OpenFold AQLab datasets..."
  
  # Download mmseqs for aqlab
  curl -L -O -J https://dev.mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
  tar -xvf mmseqs-linux-avx2.tar.gz
  rm mmseqs-linux-avx2.tar.gz
  
  # Create aqlab directory
  mkdir -p aqlab
  cd aqlab
  
  # Download the data
  mkdir -p alignment_data/alignment_dir_roda
  aws s3 cp s3://openfold/pdb/ alignment_data/alignment_dir_roda/ --recursive --no-sign-request
  
  mkdir pdb_data
  aws s3 cp s3://openfold/pdb_mmcif.zip pdb_data/ --no-sign-request
  aws s3 cp s3://openfold/duplicate_pdb_chains.txt . --no-sign-request
  unzip pdb_mmcif.zip -d pdb_data
  
  bash scripts/flatten_roda.sh alignment_data/alignment_dir_roda alignment_data/
  rm -r alignment_data/alignment_dir_roda
  
  python scripts/alignment_db_scripts/create_alignment_db_sharded.py \
      alignment_data/alignments \
      alignment_data/alignment_dbs \
      alignment_db \
      --n_shards 10 \
      --duplicate_chains_file pdb_data/duplicate_pdb_chains.txt
  
  python scripts/alignment_data_to_fasta.py \
      alignment_data/all-seqs.fasta \
      --alignment_db_index alignment_data/alignment_dbs/alignment_db.index
  
  python scripts/fasta_to_clusterfile.py \
      alignment_data/all-seqs.fasta \
      alignment_data/all-seqs_clusters-40.txt \
      ../mmseqs/bin/mmseqs \
      --seq-id 0.4
  
  aws s3 cp s3://openfold/data_caches/ pdb_data/ --recursive --no-sign-request
  
  mkdir pdb_data/data_caches
  
  python $OF_DIR/scripts/generate_mmcif_cache.py \
      pdb_data/mmcif_files \
      pdb_data/data_caches/mmcif_cache.json \
      --no_workers 16
  
  python $OF_DIR/scripts/generate_chain_data_cache.py \
      pdb_data/mmcif_files \
      pdb_data/data_caches/chain_data_cache.json \
      --cluster_file alignment_data/all-seqs_clusters-40.txt \
      --no_workers 16
  cd ..
else
  echo "Skipping OpenFold AQLab datasets"
fi

# OpenFold MLCommons
if should_download "openfold_mlcommons"; then
  echo "Downloading OpenFold MLCommons datasets..."
  
  # Create mlcommons directory
  mkdir -p mlcommons
  cd mlcommons
  bash download.sh
  cd ..
else
  echo "Skipping OpenFold MLCommons datasets"
fi

# Return to datasets directory if in OpenFold
if should_download "openfold_aqlab" || should_download "openfold_mlcommons"; then
  cd ..
fi

# GNN
if should_download "gnn"; then
  echo "Downloading GNN datasets..."
  mkdir -p GNN
  cd GNN
  python3 download.py --path ./ --dataset_type heterogeneous --dataset_size tiny --confirm-download
  cd ..
else
  echo "Skipping GNN datasets"
fi

# Llama2 fine-tuning

llama2_files=(
    ".gitattributes"
    "LICENSE.txt"
    "README.md"
    "Responsible-Use-Guide.pdf"
    "USE_POLICY.md"
    "config.json"
    "generation_config.json"
    "model-00001-of-00002.safetensors"
    "model-00002-of-00002.safetensors"
    "model.safetensors.index.json"
    "pytorch_model-00001-of-00002.bin"
    "pytorch_model-00002-of-00002.bin"
    "pytorch_model.bin.index.json"
    "special_tokens_map.json"
    "tokenizer.json"
    "tokenizer.model"
    "tokenizer_config.json"
)


if should_download "llama2_ft"; then
  echo "Downloading Llama2 fine-tuning datasets..."
  mkdir -p llama2_ft
  cd llama2_ft
  echo "Downloading Llama2 fine-tuning files..."
  for file in "${llama2_files[@]}"; do
    url="https://huggingface.co/TheBloke/Llama-2-7b-chat-GPTQ/resolve/main/$file"
    echo "Downloading $file from $url"
    curl -L -O "$url" \
         -H "Authorization: Bearer $AUTH_TOKEN"
    if [ $? -eq 0 ]; then
        echo "Successful: $file"
  done
  cd ..
else
  echo "Skipping Llama2 fine-tuning datasets"
fi

echo "Download complete for selected datasets: ${SELECTED_DATASETS[*]}"