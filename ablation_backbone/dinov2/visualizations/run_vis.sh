#!/bin/bash

# --- Configuration ---
CKPT_PATH="/home/biniyam/spectravit_reproduction/outputs/spectravit_run_02/best_model.pth"
IMAGE_PATH="/home/biniyam/Datasets/NCT-CRC-HE/CRC-VAL-HE-7K/" # Replace with a real image path or directory
OUTPUT_DIR="visualizations/outputs"

# Fix for CXXABI/libstdc++ errors
export LD_LIBRARY_PATH=$(dirname $(which python))/../lib:$LD_LIBRARY_PATH

echo "Generating Spectral Grad-CAM and Attention Maps..."
echo "Image: $IMAGE_PATH"
echo "Checkpoint: $CKPT_PATH"

python visualizations/generate_spectral_gradcam.py \
    --image_path "$IMAGE_PATH" \
    --ckpt "$CKPT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_classes 9 \
    --images_per_class 5