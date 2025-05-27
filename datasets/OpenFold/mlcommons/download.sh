#!/bin/bash

# Create a log file
LOG_FILE="download_log.txt"
echo "Download started at $(date)" > $LOG_FILE

# URLs to download
PDB_DATA_URL="https://portal.nersc.gov/cfs/m4291/openfold/pdb_data.tar"
CHECKPOINT_URL="https://portal.nersc.gov/cfs/m4291/openfold/mlperf_hpc_openfold_resumable_checkpoint_b518be46.pt"

# Function to download with wget
download_file() {
    local url=$1
    local filename=$(basename $url)
    
    echo "Starting download of $filename at $(date)" >> $LOG_FILE
    
    # Download with wget using the -c flag for resume capability
    wget -c $url -O $filename
    
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded $filename at $(date)" >> $LOG_FILE
    else
        echo "Error downloading $filename at $(date)" >> $LOG_FILE
    fi
}

# Download files
echo "Downloading PDB data file..." >> $LOG_FILE
download_file $PDB_DATA_URL

echo "Downloading checkpoint file..." >> $LOG_FILE
download_file $CHECKPOINT_URL

echo "All downloads completed at $(date)" >> $LOG_FILE
—---------------------------------------------------------------------------------
