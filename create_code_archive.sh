#!/bin/bash

# Script to create a clean zip of the Jakarta Smart City Traffic Safety codebase
# Excludes logs, model files, videos, cache files, and other non-source files
 
 
# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set the output zip name with date
TIMESTAMP=$(date +"%Y%m%d")
OUTPUT_FILE="jakarta_smart_city_code_${TIMESTAMP}.zip"

echo "Creating clean code archive: ${OUTPUT_FILE}"
echo "Working from directory: ${SCRIPT_DIR}"

# Move to the script directory to ensure relative paths work
cd "${SCRIPT_DIR}"

# Create the zip file with source code, excluding unwanted files
zip -r "${OUTPUT_FILE}" \
    . \
    -x "*.git*" \
    -x "*.mp4" \
    -x "*.mkv" \
    -x "*.pt" \
    -x "*.pth" \
    -x "*.weights" \
    -x "*.h5" \
    -x "*.log" \
    -x "logs/*" \
    -x "**/__pycache__/*" \
    -x "**/*.pyc" \
    -x "**/venv/*" \
    -x "**/.venv/*" \
    -x "**/.env/*" \
    -x "**/.ipynb_checkpoints/*" \
    -x "downloads/*" \
    -x "downloaded_videos/*" \
    -x "output/*" \
    -x "outputs/*" \
    -x "models/*" \
    -x "**/*.egg-info/*" \
    -x "build/*" \
    -x "dist/*"

if [ $? -eq 0 ]; then
    echo "Successfully created ${OUTPUT_FILE}"
    echo "Archive size: $(du -h "${OUTPUT_FILE}" | cut -f1)"
else
    echo "Error creating archive"
    exit 1
fi

echo "Done!"
