#!/bin/bash
# filepath: /Users/vinhpx/Workspaces/jakarta_analyze/scripts/run_yolo_comparison.sh

set -e  # Exit on error

# Print usage information
echo "==== Jakarta Analyze YOLO Comparison Runner ===="
echo "This script helps you run either YOLOv3, YOLOv11m-seg, or both in parallel."
echo

# Check if required models exist, download if needed
MODEL_DIR="models"
mkdir -p "$MODEL_DIR"

check_model() {
    local model_name=$1
    local model_file="$MODEL_DIR/$model_name.pt"
    
    if [ ! -f "$model_file" ]; then
        echo "Model $model_name not found, downloading..."
        python -m jakarta_analyze download-model "$model_name"
    else
        echo "Model $model_name found at $model_file"
    fi
}

# Check/download required models
check_model "yolo3u"
check_model "yolo11m-seg"

# Create output directories
mkdir -p outputs/yolov3_results
mkdir -p outputs/yolo11m_results

# Display menu for selection
echo
echo "Select which YOLO configuration to run:"
echo "1) YOLOv3 only (standard object detection)"
echo "2) YOLOv11m-seg only (with motorcycle sidewalk detection)"
echo "3) Both models in parallel (for comparison)"
echo "4) Exit"
echo

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "Running YOLOv3 pipeline..."
        python -m jakarta_analyze run-pipeline --config config_examples/pipeline.yml
        ;;
    2)
        echo "Running YOLOv11m-seg pipeline with motorcycle sidewalk detection..."
        python -m jakarta_analyze run-pipeline --config config_examples/pipeline_yolo11m_seg.yml
        ;;
    3)
        echo "Running both YOLO models in parallel for comparison..."
        python -m jakarta_analyze run-pipeline --config config_examples/pipeline_dual_yolo.yml
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac

echo "Pipeline execution complete!"
