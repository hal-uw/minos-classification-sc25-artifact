#!/bin/bash

# Script Name: collect_results.sh
# Purpose: Collect all CSV files from subdirectories to a results folder

# Define the results directory
RESULTS_DIR="results"

echo "========================================"
echo "Starting CSV results collection"
echo "========================================"

# Create results directory if it doesn't exist
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Creating results directory: $RESULTS_DIR"
    mkdir -p "$RESULTS_DIR"
else
    echo "Results directory already exists: $RESULTS_DIR"
fi

# Initialize counter
collected_count=0

echo "Searching for CSV files in all subdirectories..."

# Find all CSV files in subdirectories and copy them
find . -name "*.csv" -type f ! -path "./$RESULTS_DIR/*" | while read -r csv_file; do
    # Extract directory name and filename
    dir_name=$(dirname "$csv_file" | sed 's|^\./||')
    filename=$(basename "$csv_file")
    
    # Create new filename with directory prefix to avoid conflicts
    if [ "$dir_name" = "." ]; then
        new_filename="$filename"
    else
        # Replace slashes with underscores for nested directories
        dir_prefix=$(echo "$dir_name" | tr '/' '_')
        new_filename="${dir_prefix}_${filename}"
    fi
    
    # Copy the file
    cp "$csv_file" "$RESULTS_DIR/$new_filename"
    
    if [ $? -eq 0 ]; then
        echo "✓ Collected: $csv_file -> $RESULTS_DIR/$new_filename"
        ((collected_count++))
    else
        echo "✗ Failed to copy: $csv_file"
    fi
done

echo "========================================"
echo "CSV collection completed"
echo "========================================"
echo "Summary:"
echo "  Successfully collected: $collected_count CSV files"
echo "  Results saved to: $RESULTS_DIR/"

# List collected files
if [ $collected_count -gt 0 ]; then
    echo ""
    echo "Collected files:"
    ls -la "$RESULTS_DIR"/*.csv 2>/dev/null | while read -r line; do
        echo "  $line"
    done
else
    echo "No CSV files were found."
fi

echo ""
echo "Collection script completed."