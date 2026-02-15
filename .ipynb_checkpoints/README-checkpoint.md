# SpectraViT Reproduction Project

This repository contains a clean, reproducible implementation of **SpectraViT** (DINOv2 + Adaptive Spectral Gating) for Histopathology Image Analysis.

## Directory Structure
- `src/`: Core source code (models, data, losses, training engine).
- `scripts/`: Executable scripts for training and evaluation.
- `configs/`: Hyperparameter configurations.
- `outputs/`: Training logs and checkpoints.

## 1. Environment Setup
Create a conda environment or install requirements directly:
```bash
pip install -r requirements.txt
```

## 2. Dataset Paths
Ensure your `configs/default.yaml` points to the correct dataset.
Default assumes:
- Train/Val: `/home/jovyan/work/Datasets/NCT-CRC-HE/NCT-CRC-HE-100K-NONORM/` (Update this in yaml!)
- External: `CRC-VAL-HE-7K`

## 3. Usage Guidelines

### A. Train on CRC-100K
```bash
python scripts/train_crc.py --config configs/default.yaml --exp_name spectravit_run_01
```
Outputs (checkpoints, logs) will be in `outputs/spectravit_run_01/`.

### B. Evaluate on CRC-7K
```bash
python scripts/eval_crc7k.py --config configs/default.yaml --ckpt outputs/spectravit_run_01/best_model.pth --data_dir /path/to/CRC-VAL-HE-7K
```

### C. Robustness Evaluation
Evaluates model against 5 levels of corruption (Blur, Noise, Color Jitter).
```bash
python scripts/robustness_eval.py --config configs/default.yaml --ckpt outputs/spectravit_run_01/best_model.pth --data_dir /path/to/CRC-VAL-HE-7K
```

### D. Transfer Learning (BACH)
**Linear Probe (Frozen Backbone):**
```bash
python scripts/transfer_bach_linear.py --config configs/default.yaml --ckpt outputs/spectravit_run_01/best_model.pth --data_dir /path/to/BACH
```

**Full Finetune:**
```bash
python scripts/transfer_bach_finetune.py --config configs/default.yaml --ckpt outputs/spectravit_run_01/best_model.pth --data_dir /path/to/BACH
```


### E. Generic Inference (Unlabeled Data)
Run predictions on any folder of images (nested or flat).
```bash
python scripts/predict.py --config configs/default.yaml --ckpt outputs/spectravit_run_01/best_model.pth --data_dir /path/to/any/images
```
Outputs a CSV with class probabilities to `inference_outputs/predictions.csv`.

## 4. Smoke Tests (Verification)
To verify the wiring without full training, use the `--dry_run` flag:
```bash
python scripts/train_crc.py --dry_run
```
Or run the unit tests:
```bash
python -m unittest discover tests
```
