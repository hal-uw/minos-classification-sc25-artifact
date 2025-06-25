#!/bin/bash

# Exit immediately if a command fails
set -e

# Create results directory if it doesn't exist
mkdir -p results

echo "Starting file search and copy..."
found_any=false

# Find all files named like metric_*_0 (no extension)
while IFS= read -r file; do
    if [ -f "$file" ]; then
        # Strip directory path, keep only filename
        filename=$(basename "$file")

        # Copy to results/ while preserving filename only
        cp "$file" "results/$filename"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to copy $file"
            continue
        fi

        echo "Copied: $file -> results/$filename"
        found_any=true
    fi
done < <(find . -type f -name "metric_*_0")

# Warn if no files were found
if [ "$found_any" = false ]; then
    echo "No metric_*_0 files found."
fi

# Run the filter script on copied files
python3 filter.py --dir results --pattern "metric_*_0"
