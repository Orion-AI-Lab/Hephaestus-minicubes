#!/bin/bash

# -----------------------------------------------------------------------------
# decompress_all.sh
#
# Usage:
#   ./decompress_all.sh /path/to/downloaded_dataset
#
# Description:
#   This script extracts all .tar files in the specified directory.
# -----------------------------------------------------------------------------

# Check if a directory argument was provided
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/downloaded_dataset"
    exit 1
fi

DATASET_DIR="$1"

# Check if the provided path exists and is a directory
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Directory '$DATASET_DIR' does not exist."
    exit 1
fi

echo "Extracting all .tar files in: $DATASET_DIR"

# Loop over all .tar files and extract them
for file in "$DATASET_DIR"/*.tar; do
    if [ -f "$file" ]; then
        echo "Extracting $(basename "$file")..."
        tar -xf "$file" -C "$DATASET_DIR"
    fi
done

echo "All .tar files have been decompressed."
