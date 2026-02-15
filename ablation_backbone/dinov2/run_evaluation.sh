#!/bin/bash

# --- Configuration ---
# Update these paths to match your actual file locations
CKPT_PATH="/home/biniyam/spectravit_reproduction/outputs/spectravit_run_02/best_model.pth"
DATA_DIR="/home/biniyam/Datasets/NCT-CRC-HE/CRC-VAL-HE-7K/" # Must be in ImageFolder format (e.g., data/class_name/image.png)
OUTPUT_DIR="outputs"
PLOT_DIR="evaluation_plots"
MODEL_TYPE="spectravit" # Options: 'spectravit', 'spectragate', or 'baseline'

# Fix for CXXABI/libstdc++ errors by prioritizing Conda environment libraries
export LD_LIBRARY_PATH=$(dirname $(which python))/../lib:$LD_LIBRARY_PATH

# --- 1. Run Inference ---
echo "----------------------------------------------------------------"
echo "Running inference with model: $MODEL_TYPE"
echo "Data Directory: $DATA_DIR"
echo "----------------------------------------------------------------"

python inference.py \
    --model "$MODEL_TYPE" \
    --ckpt "$CKPT_PATH" \
    --data "$DATA_DIR" \
    --output "$OUTPUT_DIR"

# --- 2. Generate Plots ---
NPZ_FILE="${OUTPUT_DIR}/${MODEL_TYPE}_inference_results.npz"

echo "Generating plots from $NPZ_FILE..."
python generate_plots.py \
    --input_npz "$NPZ_FILE" \
    --output_dir "$PLOT_DIR" \
    --tsne_samples 2000