#!/bin/bash

# Exit on error
set -e

# Create results directory
mkdir -p results

# Find and copy files
echo "Starting file search and copy..."
found=false

# Use find command to recursively search for files
# Using two patterns:
# 1. metrics_* (original pattern)
find . -type f \( -name "metrics_*" \) | while read file; do
    if [ -f "$file" ]; then
        cp "$file" results/
        # Check if the file was copied successfully
        if [ $? -ne 0 ]; then
            echo "Error: Failed to copy $file"
            continue
        fi
        # then delete the file
        rm "$file"
        echo "Copied: $file -> results/"
        found=true
    fi
done
# 2. call the filter python script
python3 filter.py --dir results --pattern "metrics_*"